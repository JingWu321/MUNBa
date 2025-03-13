import copy
import os
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import utils

import clip
from unlearn import MUNBa, FT, GA, SalUn, SHs


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
    # print(train_loader_full.dataset.breed_to_idx)
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    # [2] prepare model
    model, preprocess = clip.load(args.arch, device=device)
    model.eval()
    # print(model)

    # prompts = [f"an image of a {label}" for label in class_name]
    prompts = [f"A photo of a {label}, a type of pet" for label in class_name]
    print(prompts, len(class_name))
    texts = clip.tokenize(prompts).to(device)
    logit_scale = 100
    criterion = nn.CrossEntropyLoss()
    evaluation_result = {}

    # # Evaluate before unlearning
    # accuracy_origin = {}
    # for name, loader in unlearn_data_loaders.items():
    #     print(name)
    #     utils.dataset_convert_to_test(loader.dataset, args)
    #     val_acc = utils.validate(loader, texts, logit_scale, model, criterion, device, args)
    #     accuracy_origin[name] = val_acc
    #     print(f"Before unlearning, {name} acc: {val_acc}")
    # evaluation_result["accuracy_origin"] = accuracy_origin
    # utils.save_checkpoint(evaluation_result, False, args.save_dir, args.unlearn, filename="eval_result.pth.tar")

    # [3] unlearn
    if args.unlearn == "FT":
        FT.Finetune(texts, unlearn_data_loaders, model, args, class_name, with_l1=False)
    elif args.unlearn == "GA":
        GA.GradientAscent(texts, unlearn_data_loaders, model, args, class_name)
    elif args.unlearn == "l1_sparse":
        FT.Finetune(texts, unlearn_data_loaders, model, args, class_name, with_l1=True)
    elif args.unlearn == "SalUn":
        mask = torch.load(args.mask)
        SalUn.SaliencyUnlearn(texts, unlearn_data_loaders, model, args, class_name, mask=mask)
    elif args.unlearn == "SHs":
        SHs.Scissorhands(texts, unlearn_data_loaders, model, args, class_name)
    elif args.unlearn == "MUNBa":
        MUNBa.munba(texts, unlearn_data_loaders, model, args, class_name)
    else:
        raise ValueError(f"unlearn method {args.unlearn} not implemented")

    utils.save_checkpoint(model.state_dict(), False, args.save_dir, args.unlearn)

    # Evaluate after unlearning
    accuracy_unlearn = {}
    for name, loader in unlearn_data_loaders.items():
        print(name)
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = utils.validate(loader, texts, logit_scale, model, criterion, device, args)
        accuracy_unlearn[name] = val_acc
        print(f"After unlearning, {name} acc: {val_acc}")
    evaluation_result["accuracy_unlearn"] = accuracy_unlearn
    utils.save_checkpoint(evaluation_result, False, args.save_dir, args.unlearn, filename="eval_result.pth.tar")


if __name__ == "__main__":
    main()

