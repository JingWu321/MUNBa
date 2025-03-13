import time
import gc

import numpy as np
import torch
import torch.nn as nn
import utils
from itertools import zip_longest
import cvxpy as cp

import clip


def l1_regularization(parameters):
    params_vec = []
    for param in parameters:
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def _stop_criteria(gtg, alpha_t, alpha_param, prvs_alpha_param):
    return (
        (alpha_param.value is None)
        or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
        or (
            np.linalg.norm(alpha_param.value - prvs_alpha_param.value)
            < 1e-3
        )
    )


def return_weights(grads, prvs_alpha, G_param, normalization_factor_param,
                   alpha_param, prvs_alpha_param, prob):
    G = torch.stack(tuple(v for v in grads.values()))
    GTG = torch.mm(G, G.t())
    normalization_factor = (
        torch.norm(GTG).detach().cpu().numpy().reshape((1,)) + 1e-6
        )
    if (np.isnan(normalization_factor) | np.isinf(normalization_factor)).any():
        normalization_factor = np.array([1.0])
    GTG = GTG / normalization_factor.item()
    gtg = GTG.cpu().detach().numpy()
    G_param.value = gtg
    normalization_factor_param.value = normalization_factor

    optim_niter=100
    alpha_t = prvs_alpha
    for _ in range(optim_niter):
        try:
            alpha_param.value = alpha_t
            prvs_alpha_param.value = alpha_t
            # try:
            prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
        except:
            alpha_param.value = prvs_alpha_param.value

        if _stop_criteria(gtg, alpha_t, alpha_param, prvs_alpha_param):
            break

        alpha_t = alpha_param.value
    if alpha_t is not None and not (np.isnan(alpha_t) | np.isinf(alpha_t)).any():
        return alpha_t
    else:
        return prvs_alpha


def munba(texts, data_loaders, model, args, class_name):

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")

    # n_tasks = 2 # K
    # prvs_alpha = np.ones(n_tasks, dtype=np.float32) # alpha from iteration

    #### Convex Optimization Problem (bargaining game) Initialization ####
    n_tasks = 2 # K
    init_gtg = np.eye(n_tasks) # G^T G: gradient matrix product, shape: [K, K]
    G_param = cp.Parameter(shape=(n_tasks, n_tasks), value=init_gtg) # will be updated in-loop with the current GTG
    normalization_factor_param = cp.Parameter( shape=(1,), value=np.array([1.0])) # will be updated in-loop with torch.norm(GTG).detach().cpu().numpy().reshape((1,))
    alpha_param = cp.Variable(shape=(n_tasks,), nonneg=True) # current alpha, shape: [K,]
    prvs_alpha = np.ones(n_tasks, dtype=np.float32) # alpha from iteration
    prvs_alpha_param = cp.Parameter(shape=(n_tasks,), value=prvs_alpha) # shape: [K,]

    # First-order approximation of Phi_alpha using Phi_alpha_(tao)
    G_prvs_alpha = G_param @ prvs_alpha_param
    prvs_phi_tag = 1 / prvs_alpha_param + (1 / G_prvs_alpha) @ G_param
    phi_alpha = prvs_phi_tag @ (alpha_param - prvs_alpha_param)

    # Beta(alpha)
    G_alpha = G_param @ alpha_param

    # Constraint: For any i, Phi_i_alpha >= 0
    constraint = []
    for i in range(n_tasks):
        constraint.append(
            -cp.log(alpha_param[i] * normalization_factor_param)
            - cp.log(G_alpha[i])
            <= 0
        )

    # Objective: Minimize sum(Phi_alpha) + Phi_alpha / normalization_factor_param
    obj = cp.Minimize(
        cp.sum(G_alpha) + phi_alpha / normalization_factor_param
    )
    prob = cp.Problem(obj, constraint)
    print("Convex optimization problem initialized.")
    # #####################################################

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # choose parameters to train based on train_method
    parameters = []
    # Freeze all parameters and then Unfreeze the transformer in the text/visual encoder
    for param in model.parameters():
        param.requires_grad = False
    if args.mode == "text":
        print("Unfreezing text encoder")
        for name, param in model.transformer.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                parameters.append(param)
    elif args.mode == "image":
        print("Unfreezing visual encoder")
        for name, param in model.visual.transformer.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                parameters.append(param)
    elif args.mode == "all":
        print("Unfreezing all parameters")
        for name, param in model.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                parameters.append(param)

    optimizer = torch.optim.SGD(parameters, args.unlearn_lr,momentum=args.momentum, weight_decay=args.weight_decay,)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_u = utils.AverageMeter()
    loader_len = max(len(forget_loader), len(retain_loader))
    logit_scale = 100

    for epoch in range(0, args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        i = 0
        start = time.time()
        for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
            i += 1
            if (data_r is None) and (data_u is None):
                break
            elif (data_u is None) and (data_r is not None):

                image_r, target_r = data_r
                image_r = image_r.to(device)
                target_r = target_r.to(device)

                optimizer.zero_grad()
                if args.mode == "text":
                    with torch.no_grad():
                        image_features = model.encode_image(image_r)  # 2bsx512
                    text_features = model.encode_text(texts) # Cx512
                elif args.mode == "image":
                    image_features = model.encode_image(image_r)  # 2bsx512
                    with torch.no_grad():
                        text_features = model.encode_text(texts) # Cx512
                elif args.mode == "all":
                    image_features = model.encode_image(image_r)  # 2bsx512
                    text_features = model.encode_text(texts) # Cx512

                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()

                loss = criterion(cosine_similarity, target_r)
                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / (args.unlearn_epochs))
                    loss = loss + current_alpha * l1_regularization(parameters)
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                with torch.no_grad():
                    # preds = torch.argmax(cosine_similarity, dim=-1).cpu().numpy()
                    prec_r = utils.accuracy(cosine_similarity, target_r)[0]
                    loss = loss.float()
                    losses.update(loss.item(), image_r.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    print(f'Batch: {i+1:4d}, prec_r: {top1.val:.3f} ({top1.avg:.3f}),loss: {loss:.4f}')

            else:
                image_r, target_r = data_r
                image_u, target_u = data_u
                image_r, target_r = image_r.to(device), target_r.to(device)
                image_u, target_u = image_u.to(device), target_u.to(device)
                # assign a random label to image_u
                target_u_rl = torch.randint(0, args.num_classes, target_u.shape, device=device)

                # concatenate images
                images = torch.cat((image_r, image_u), dim=0)
                bs = image_r.size(0)

                optimizer.zero_grad()
                if args.mode == "text":
                    with torch.no_grad():
                        image_features = model.encode_image(images)  # 2bsx512
                    text_features = model.encode_text(texts) # Cx512
                elif args.mode == "image":
                    image_features = model.encode_image(images)  # 2bsx512
                    with torch.no_grad():
                        text_features = model.encode_text(texts) # Cx512
                elif args.mode == "all":
                    image_features = model.encode_image(images)  # 2bsx512
                    text_features = model.encode_text(texts) # Cx512

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features_r = image_features[:bs] / image_features[:bs].norm(dim=-1, keepdim=True)
                cosine_similarity_r = logit_scale * image_features_r @ text_features.t()
                loss_r = criterion(cosine_similarity_r, target_r)

                image_features_u = image_features[bs:] / image_features[bs:].norm(dim=-1, keepdim=True)
                cosine_similarity_u = logit_scale * image_features_u @ text_features.t()
                loss_u = args.beta * criterion(cosine_similarity_u, target_u_rl)

                #####################################################
                # compute gradient for each task
                grads = {}
                for task, loss in zip([0, 1], [loss_r, loss_u]):
                    optimizer.zero_grad()
                    grad = torch.autograd.grad(loss, parameters, retain_graph=True)
                    grads[task] = torch.cat([torch.flatten(g.detach()) for g in grad])

                # # ############# [1] Choose to use the iterative solution
                # prvs_alpha = return_weights(grads, prvs_alpha, G_param, normalization_factor_param,
                #                             alpha_param, prvs_alpha_param, prob)
                # if np.all(prvs_alpha == 1): # Bargaining failed
                #     # continue
                #     loss_final = loss_r + loss_u * 0.1
                # else:
                #     loss_final = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]

                ############ [2] Choose to use the closed-form solution
                g1 = torch.dot(grads[0], grads[0])
                g2 = torch.dot(grads[0], grads[1])
                g3 = torch.dot(grads[1], grads[1])
                prvs_alpha[0] = torch.sqrt( (g1*g3 - g2*torch.sqrt(g1*g3)) / (g1*g1*g3 - g1*g2*g2 + 1e-8) )
                prvs_alpha[1] = (1 - g1 * prvs_alpha[0] * prvs_alpha[0]) / (g2*prvs_alpha[0] + 1e-8)

                if prvs_alpha[0] > 0 and prvs_alpha[1] > 0: # Bargaining succeeded
                    loss_final = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]
                else:
                    continue
                #####################################################

                optimizer.zero_grad()
                if args.with_l1:
                    current_alpha = args.alpha * (1 - epoch / (args.unlearn_epochs))
                    loss_final = loss_final + current_alpha * l1_regularization(parameters)
                loss_final.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                # measure accuracy and record loss
                with torch.no_grad():
                    # preds_r = torch.argmax(cosine_similarity_r, dim=-1).cpu().numpy()
                    # preds_u = torch.argmax(cosine_similarity_u, dim=-1).cpu().numpy()
                    prec_r = utils.accuracy(cosine_similarity_r, target_r)[0]
                    prec_u = utils.accuracy(cosine_similarity_u, target_u)[0]
                    loss = loss_final.float()
                    losses.update(loss.item(), image_r.size(0) + image_u.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    top1_u.update(prec_u.item(), image_u.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 2 == 0:
                    print(f'Batch: {i+1:4d}, prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f}), loss_u: {args.lam * loss_u:.4f}, loss_r: {loss_r:.4f}, prvs_alpha: {prvs_alpha}')

            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))
