import time
import copy
import math
import gc

import torch
import torch.nn as nn
from itertools import zip_longest

import utils
import clip


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def re_init_weights(shape, device):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
        mask = torch.squeeze(mask, 1)
    else:
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    return mask


def create_dense_mask(net, device, value=1):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def snip(model, dataloader, texts, logit_scale, sparsity, prune_num, device, mode='text'):
    criterion = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False
    grads = []
    weights = []
    if mode == "text":
        print("Unfreezing text encoder")
        for name, param in model.transformer.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                grads.append(torch.zeros_like(param))
                weights.append(param)

    elif mode == "image":
        print("Unfreezing visual encoder")
        # j = 0
        for name, param in model.visual.transformer.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                grads.append(torch.zeros_like(param))
                weights.append(param)
            # j += 1
        # print(f"number of attn layers: {j}")

    elif mode == "all":
        print("Unfreezing all parameters")
        for name, param in model.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                grads.append(torch.zeros_like(param))
                weights.append(param)

    # compute grads
    for ii in range(prune_num):
        image, target = next(iter(dataloader))
        image, target = image.to(device), target.to(device)

        if mode == "text":
            with torch.no_grad():
                image_features = model.encode_image(image)  # bsx512
            text_features = model.encode_text(texts) # Cx512
        elif mode == "image":
            image_features = model.encode_image(image)  # bsx512
            with torch.no_grad():
                text_features = model.encode_text(texts) # Cx512
        elif mode == "all":
            image_features = model.encode_image(image)  # bsx512
            text_features = model.encode_text(texts) # Cx512

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = logit_scale * image_features @ text_features.t()

        loss = criterion(cosine_similarity, target)
        model.zero_grad()
        loss.backward()

        j = 0
        with torch.no_grad():
            if mode == "text":
                for name, param in model.transformer.named_parameters():
                    if (param.grad is not None) and ('attn' in name):
                        grads[j] += (param.grad.data).abs()
                        j += 1
            elif mode == "image":
                for name, param in model.visual.transformer.named_parameters():
                    if (param.grad is not None) and ('attn' in name):
                        grads[j] += (param.grad.data).abs()
                        j += 1
            elif mode == "all":
                for name, param in model.named_parameters():
                    if (param.grad is not None) and ('attn' in name):
                        grads[j] += (param.grad.data).abs()
                        j += 1
        torch.cuda.empty_cache()
        gc.collect()


    # compute saliences to get the threshold
    mask_ = create_dense_mask(copy.deepcopy(model), device, value=1)
    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for weight, grad in zip(weights, grads)]
        saliences = [saliences.view(-1).cpu() for saliences in abs_saliences]
        saliences = torch.cat(saliences).to(torch.float32)
        # threshold = np.percentile(saliences, sparsity * 100) # kx100-th percentile
        threshold = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0]) # k-th smallest value
        # if (threshold >= saliences.max() - 1e-12) or (threshold <= saliences.min() + 1e-12):
        #     threshold = (saliences.max() - saliences.min()) / 2.

        # get mask to prune the weights
        if mode == "text":
            attn_params = [param for name, param in mask_.transformer.named_parameters() if 'attn' in name]
        elif mode == "image":
            attn_params = [param for name, param in mask_.visual.transformer.named_parameters() if 'attn' in name]
        elif mode == "all":
            attn_params = [param for name, param in mask_.named_parameters() if 'attn' in name]
        for j, param in enumerate(attn_params):
            indx = (abs_saliences[j] > threshold) # prune for forget data
            param.data[indx] = 0

        # update the weights of the original network with the mask
        if mode == "text":
            for (name, param), (m_param) in zip(model.transformer.named_parameters(), mask_.transformer.parameters()):
                if 'attn' in name:
                    if ('weight' in name):
                        re_init_param = re_init_weights(param.data.shape, device)
                    elif ('bias' in name):
                        re_init_param = torch.nn.init.zeros_(torch.empty(param.data.shape, device=device))
                    data_type = param.dtype
                    param.data = (param.data * m_param.data + re_init_param.data * (1 - m_param.data)).to(data_type) # convert back to original data type

        elif mode == "image":
            for (name, param), (m_param) in zip(model.visual.transformer.named_parameters(), mask_.visual.transformer.parameters()):
                if ('attn' in name):
                # if ('attn' in name) and ('out_proj' in name):
                    if ('weight' in name):
                        re_init_param = re_init_weights(param.data.shape, device)
                        data_type = param.dtype
                        param.data = (param.data * m_param.data + re_init_param.data * (1 - m_param.data)).to(data_type)
                    elif ('bias' in name):
                        re_init_param = torch.nn.init.zeros_(torch.empty(param.data.shape, device=device))
                        data_type = param.dtype
                        param.data = (param.data * m_param.data + re_init_param.data * (1 - m_param.data)).to(data_type)

        elif mode == "all":
            for (name, param), (m_param) in zip(model.named_parameters(), mask_.parameters()):
                if 'attn' in name:
                    if ('weight' in name):
                        re_init_param = re_init_weights(param.data.shape, device)
                    elif ('bias' in name):
                        re_init_param = torch.nn.init.zeros_(torch.empty(param.data.shape, device=device))
                    data_type = param.dtype
                    param.data = (param.data * m_param.data + re_init_param.data * (1 - m_param.data)).to(data_type)
                    # if param.dtype == torch.float32:
                    #     param.data = param.data.to(torch.float16)

    return model


def Scissorhands(texts, data_loaders, model, args, class_name):

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")
    logit_scale = 100

    # prune via snip
    model = snip(model, forget_loader, texts, logit_scale, args.sparsity, args.prune_num, device, args.mode)

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # choose parameters to train based on train_method
    parameters = []
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
    prompts = [f"an image of a {label}" for label in class_name]
    texts = clip.tokenize(prompts).to(device)

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
                # # assign a random label to image_u
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
                # print(image_features.size(), text_features.size())
                image_features_r = image_features[:bs] / image_features[:bs].norm(dim=-1, keepdim=True)
                cosine_similarity_r = logit_scale * image_features_r @ text_features.t()
                loss_r = criterion(cosine_similarity_r, target_r)

                image_features_u = image_features[bs:] / image_features[bs:].norm(dim=-1, keepdim=True)
                cosine_similarity_u = logit_scale * image_features_u @ text_features.t()
                loss_u = criterion(cosine_similarity_u, target_u_rl)
                # loss_u = criterion(cosine_similarity_u, target_u)

                loss = loss_r + args.lam * loss_u
                # loss = loss_r - args.lam * loss_u
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                # measure accuracy and record loss
                with torch.no_grad():
                    prec_r = utils.accuracy(cosine_similarity_r, target_r)[0]
                    prec_u = utils.accuracy(cosine_similarity_u, target_u)[0]
                    loss = loss.float()
                    losses.update(loss.item(), image_r.size(0) + image_u.size(0))
                    top1.update(prec_r.item(), image_r.size(0))
                    top1_u.update(prec_u.item(), image_u.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0:
                    print(f'Batch: {i+1:4d}, prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f}), loss_u: {args.lam * loss_u:.4f}, loss_r: {loss_r:.4f}')

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



