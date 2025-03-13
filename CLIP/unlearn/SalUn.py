import time
import gc
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

import utils
import clip


def SaliencyUnlearn(texts, data_loaders, model, args, class_name, mask=None):

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")

    # forget_dataset = deepcopy(forget_loader.dataset)
    # # print(sorted(set(forget_dataset.targets)))
    # forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
    # # print(sorted(set(forget_dataset.targets)))
    # retain_dataset = retain_loader.dataset
    # train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    # Freeze all parameters and then Unfreeze the transformer in the text/visual encoder
    parameters = []
    for param in model.parameters():
        param.requires_grad = False
    if args.mode == "text":
        print("Unfreezing text encoder")
        for param in model.transformer.parameters():
            param.requires_grad = True
            parameters.append(param)
    elif args.mode == "image":
        print("Unfreezing visual encoder")
        for param in model.visual.parameters():
            param.requires_grad = True
            parameters.append(param)
    elif args.mode == "all":
        print("Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True
            parameters.append(param)

    optimizer = torch.optim.SGD(parameters, args.unlearn_lr,momentum=args.momentum, weight_decay=args.weight_decay,)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    # loader_len = len(train_loader)
    loader_len = len(retain_loader)
    # # tokenize
    # prompts = [f"an image of a {label}" for label in class_name]
    # texts = clip.tokenize(prompts).to(device)
    logit_scale = 100

    for epoch in range(0, args.unlearn_epochs):
        start_time = time.time()
        model.train()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        start = time.time()
        for i, data in enumerate(forget_loader):
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            targets_rl = torch.randint(0, args.num_classes, targets.shape, device=device)

            optimizer.zero_grad()
            if args.mode == "text":
                with torch.no_grad():
                    image_features = model.encode_image(images)  # bsx512
                text_features = model.encode_text(texts) # Cx512
            elif args.mode == "image":
                image_features = model.encode_image(images)  # bsx512
                with torch.no_grad():
                    text_features = model.encode_text(texts) # Cx512
            elif args.mode == "all":
                image_features = model.encode_image(images)  # bsx512
                text_features = model.encode_text(texts) # Cx512

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()

            loss = criterion(cosine_similarity, targets_rl)

            optimizer.zero_grad()
            loss.backward()
            if mask:
                if args.mode == "text":
                    for name, param in model.transformer.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
                elif args.mode == "image":
                    for name, param in model.visual.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.visual.parameters(), 1.0)
                elif args.mode == "all":
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for i, data in enumerate(retain_loader):
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            if args.mode == "text":
                with torch.no_grad():
                    image_features = model.encode_image(images)  # bsx512
                text_features = model.encode_text(texts) # Cx512
            elif args.mode == "image":
                image_features = model.encode_image(images)  # bsx512
                with torch.no_grad():
                    text_features = model.encode_text(texts) # Cx512
            elif args.mode == "all":
                image_features = model.encode_image(images)  # bsx512
                text_features = model.encode_text(texts) # Cx512

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = logit_scale * image_features @ text_features.t()

            loss = criterion(cosine_similarity, targets)

            optimizer.zero_grad()
            loss.backward()
            if mask:
                if args.mode == "text":
                    for name, param in model.transformer.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
                elif args.mode == "image":
                    for name, param in model.visual.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.visual.parameters(), 1.0)
                elif args.mode == "all":
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # measure accuracy and record loss
            with torch.no_grad():
                prec = utils.accuracy(cosine_similarity, targets)[0]
                loss = loss.float()
                losses.update(loss.item(), images.size(0))
                top1.update(prec.item(), images.size(0))
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()


        # for i, data in enumerate(train_loader):
        #     images, targets = data
        #     images, targets = images.to(device), targets.to(device)

        #     optimizer.zero_grad()
        #     if args.mode == "text":
        #         with torch.no_grad():
        #             image_features = model.encode_image(images)  # bsx512
        #         text_features = model.encode_text(texts) # Cx512
        #     elif args.mode == "image":
        #         image_features = model.encode_image(images)  # bsx512
        #         with torch.no_grad():
        #             text_features = model.encode_text(texts) # Cx512
        #     elif args.mode == "all":
        #         image_features = model.encode_image(images)  # bsx512
        #         text_features = model.encode_text(texts) # Cx512

        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #     cosine_similarity = logit_scale * image_features @ text_features.t()

        #     loss = criterion(cosine_similarity, targets)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     if mask:
        #         if args.mode == "text":
        #             for name, param in model.transformer.named_parameters():
        #                 if param.grad is not None:
        #                     param.grad *= mask[name]
        #             # nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
        #         elif args.mode == "image":
        #             for name, param in model.visual.named_parameters():
        #                 if param.grad is not None:
        #                     param.grad *= mask[name]
        #             # nn.utils.clip_grad_norm_(model.visual.parameters(), 1.0)
        #         elif args.mode == "all":
        #             for name, param in model.named_parameters():
        #                 if param.grad is not None:
        #                     param.grad *= mask[name]
        #             # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #     optimizer.step()

        #     # measure accuracy and record loss
        #     with torch.no_grad():
        #         prec = utils.accuracy(cosine_similarity, targets)[0]
        #         loss = loss.float()
        #         losses.update(loss.item(), images.size(0))
        #         top1.update(prec.item(), images.size(0))
        #         torch.cuda.empty_cache()
        #         gc.collect()

        #     if (i + 1) % args.print_freq == 0:
        #        end = time.time()
        #        print('Epoch: [{0}][{1}/{2}]\t'
        #              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #              'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
        #              'Time {3:.2f}'.format(
        #                  epoch, i, loader_len, end-start, loss=losses, top1=top1))
        #        start = time.time()

        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))
