import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")
    ##################################### Dataset #################################################
    parser.add_argument("--data", type=str, default="/data/datasets/oxford_pets", help="location of the data corpus")
    parser.add_argument("--dataset", type=str, default="pets", help="dataset")
    parser.add_argument("--input_size", type=int, default=224, help="size of input images")
    parser.add_argument("--data_dir", type=str, default="/data/datasets/oxford_pets", help="dir to oxford_pets dataset")
    parser.add_argument("--num_classes", type=int, default=37)

    ##################################### Architecture ############################################
    parser.add_argument("--arch", type=str, default="ViT-B/32", help="model architecture")  # ViT-L/14

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--workers", type=int, default=4, help="number of workers in dataloader")
    parser.add_argument("--save_dir", help="The directory used to save the trained models", default=None, type=str)

    ##################################### Training setting #################################################
    parser.add_argument("--mode", default='text', type=str, help="finetune mode, text, image, or all")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--epochs", default=182, type=int, help="number of total epochs to run")
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument("--no-aug", action="store_true", default=False, help="No augmentation in training dataset (transformation).")
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")

    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn", type=str, default="FT", help="method to unlearn")
    parser.add_argument("--unlearn_lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--unlearn_epochs", default=10, type=int, help="number of total epochs for unlearn to run")
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### SHs setting #################################################
    parser.add_argument("--sparsity", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--project", action="store_true", default=False)
    parser.add_argument("--memory_num", type=int, default=10)
    parser.add_argument("--prune_num", type=int, default=1)
    parser.add_argument("--shrink", action="store_true", default=False)

    parser.add_argument("--with_l1", action="store_true", default=False)
    ##################################### MUNBa setting #################################################
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--skip", action="store_true", default=False)
    return parser.parse_args()

