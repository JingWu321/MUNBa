import copy
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import utils
import clip


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    # [1] prepare dataset
    (
        train_loader_full,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        class_name
    ) = utils.setup_dataset(args)
    retain_dataset = retain_loader.dataset
    forget_dataset = forget_loader.dataset

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    # [2] prepare model
    model, preprocess = clip.load(args.arch, device=device)
    model.eval()

    prompts = [f"an image of a {label}" for label in class_name]
    print(prompts, len(class_name))
    texts = clip.tokenize(prompts).to(device)
    logit_scale = 100
    forget_loader = unlearn_data_loaders["forget"]
    model.eval()

    gradients = {}
    criterion = nn.CrossEntropyLoss()
    for param in model.parameters():
        param.requires_grad = False
    if args.mode == "text":
        print("Unfreezing text encoder")
        for name, param in model.transformer.named_parameters():
            param.requires_grad = True
            gradients[name] = 0
        optimizer = torch.optim.SGD(model.transformer.parameters(), args.unlearn_lr,momentum=args.momentum, weight_decay=args.weight_decay,)
    elif args.mode == "image":
        print("Unfreezing visual encoder")
        for name, param in model.visual.named_parameters():
            param.requires_grad = True
            gradients[name] = 0
        optimizer = torch.optim.SGD(model.visual.parameters(), args.unlearn_lr,momentum=args.momentum, weight_decay=args.weight_decay,)
    elif args.mode == "all":
        print("Unfreezing all parameters")
        for name, param in model.named_parameters():
            param.requires_grad = True
            gradients[name] = 0
        optimizer = torch.optim.SGD(model.parameters(), args.unlearn_lr,momentum=args.momentum, weight_decay=args.weight_decay,)

    for i, (image, target) in enumerate(forget_loader):
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        if args.mode == "text":
            with torch.no_grad():
                image_features = model.encode_image(image)  # bsx512
            text_features = model.encode_text(texts) # Cx512
        elif args.mode == "image":
            image_features = model.encode_image(image)  # bsx512
            with torch.no_grad():
                text_features = model.encode_text(texts) # Cx512
        elif args.mode == "all":
            image_features = model.encode_image(image)  # bsx512
            text_features = model.encode_text(texts) # Cx512

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = logit_scale * image_features @ text_features.t()

        loss = -criterion(cosine_similarity, target)
        loss.backward()

        with torch.no_grad():
            if args.mode == "text":
                for name, param in model.transformer.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad
            elif args.mode == "image":
                for name, param in model.visual.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad
            elif args.mode == "all":
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])
    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # threshold_list = [0.2, 0.8, 0.5]
    threshold_list = [0.5]

    for i in threshold_list:
        print(i)
        sorted_dict_positions = {}
        hard_dict = {}

        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * i)
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(args.save_dir, "with_{}.pt".format(i)))


if __name__ == "__main__":
    main()




