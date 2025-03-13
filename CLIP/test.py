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
from loadData.dataset import standfordCars_dataloaders, imagenet_dataloaders



# zero-shot prediction
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
    print("Zero-shot prediction")
    if args.dataset == 'stanfordCars':
        _, testloader, class_name = standfordCars_dataloaders(
            batch_size=args.batch_size, data_dir=args.data,
            num_workers=args.workers, seed=args.seed)
        name = ['StanfordCars']
        exclude_classes, class_map = None, None
        prompts = [f"A photo of a {label}, a type of pet" for label in class_name]
    elif args.dataset == 'imagenet':
        _, testloader, class_name, exclude_classes, class_map = imagenet_dataloaders(
            batch_size=args.batch_size, data_dir=args.data,
            num_workers=args.workers, seed=args.seed)
        name = ['ImageNet']
        prompts = [f"A photo of a {label}" for label in class_name]
        # imagenet_templates = [
        #     'itap of a {}.',
        #     'a bad photo of the {}.',
        #     'a origami {}.',
        #     'a photo of the large {}.',
        #     'a {} in a video game.',
        #     'art of the {}.',
        #     'a photo of the small {}.'
        # ]
        # prompts = [template.format(label) for template in imagenet_templates for label in class_name]

    # print(prompts, len(class_name))
    texts = clip.tokenize(prompts).to(device)
    logit_scale = 100
    criterion = nn.CrossEntropyLoss()
    evaluation_result = {}

    # [2] Load original model
    model, preprocess = clip.load(args.arch, device=device)
    model.eval()

    if not args.skip:
        # Evaluate before unlearning
        utils.dataset_convert_to_test(testloader.dataset, args)
        # if args.dataset == 'imagenet':
        #     val_acc_top1, val_acc_top5 = utils.validate(testloader, texts, logit_scale, model, criterion, device, args, exclude_classes, class_map)
        #     print(f"Zero-shot prediction before unlearning, {name} acc: {val_acc_top1}, {val_acc_top5}")
        #     evaluation_result["accuracy_origin"] = val_acc_top1
        #     evaluation_result["accuracy_origin_top5"] = val_acc_top5
        # else:
        val_acc = utils.validate(testloader, texts, logit_scale, model, criterion, device, args, exclude_classes, class_map)
        print(f"Zero-shot prediction before unlearning, {name} acc: {val_acc}")
        evaluation_result["accuracy_origin"] = val_acc
        utils.save_checkpoint(evaluation_result, False, args.save_dir+'/'+args.dataset, args.unlearn, filename="eval_result_zs.pth.tar")

    # [3] Load unlearned model
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn, filename="checkpoint.pth.tar")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Evaluate after unlearning
    utils.dataset_convert_to_test(testloader.dataset, args)
    # if args.dataset == 'imagenet':
    #     val_acc_top1, val_acc_top5 = utils.validate(testloader, texts, logit_scale, model, criterion, device, args, exclude_classes, class_map)
    #     print(f"Zero-shot prediction after unlearning, {name} acc: {val_acc_top1}, {val_acc_top5}")
    #     evaluation_result["accuracy_unlearn"] = val_acc_top1
    #     evaluation_result["accuracy_unlearn_top5"] = val_acc_top5
    # else:
    val_acc = utils.validate(testloader, texts, logit_scale, model, criterion, device, args, exclude_classes, class_map)
    print(f"Zero-shot prediction after unlearning, {name} acc: {val_acc}")
    evaluation_result["accuracy_unlearn"] = val_acc

    utils.save_checkpoint(evaluation_result, False, args.save_dir+'/'+args.dataset, args.unlearn, filename="eval_result_zs.pth.tar")


if __name__ == "__main__":
    main()


