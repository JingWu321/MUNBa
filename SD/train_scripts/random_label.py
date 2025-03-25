import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import setup_forget_data, setup_model, setup_remain_data
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm
import time

# import copy
# from myGFK import GFK


def certain_label(
    class_to_forget,
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)

    # # GFK
    # if args.GFK:
    #     proxy_model = copy.deepcopy(model).to(device)
    #     proxy_model.eval()
    #     gfk = GFK(device=device, dim=args.dim)

    # set model to train
    model.train()
    losses = []

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                print(name)
                parameters.append(param)

    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
        # name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}-gamma_{args.gamma}-lam1_{args.lam1}-lam2_{args.lam2}-dim_{args.dim}"
    else:
        name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
        # name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}-gamma_{args.gamma}-lam1_{args.lam1}-lam2_{args.lam2}-dim_{args.dim}"

    # TRAINING CODE
    step = 0
    start_time = time.time()
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time_1:
            model.train()

            for i, (images, labels) in enumerate(forget_dl):
                optimizer.zero_grad()

                forget_images, forget_labels = next(iter(forget_dl))
                remain_images, remain_labels = next(iter(remain_dl))

                forget_prompts = [descriptions[label] for label in forget_labels]

                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]
                remain_prompts = [descriptions[label] for label in remain_labels]
                # print(forget_prompts, pseudo_prompts, remain_prompts)

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model.shared_step(remain_batch)[0]
                # remain_loss, _, fms_r = model.shared_step(remain_batch, isFeature=True)

                # forget stage
                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }

                pseudo_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": pseudo_prompts,
                }

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(
                    0,
                    model.num_timesteps,
                    (forget_input.shape[0],),
                    device=model.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model.device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                # forget_out, fms_u = model.apply_model(forget_noisy, t, forget_emb, isFeature=True)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()  # [B, 4, 64, 64]

                # # GFK
                # if args.GFK:
                #     lgr = 0
                #     lgf = 0
                #     bs = forget_noisy.size(0)
                #     with torch.no_grad():
                #         proxy_fms_r = proxy_model.shared_step(remain_batch, isFeature=True)[2] # [B, 320, 64, 64] or [B, 1280, 8, 8]
                #         _, proxy_fms_u = proxy_model.apply_model(forget_noisy, t, forget_emb, isFeature=True)
                #         # print(proxy_fms_r.shape, proxy_fms_u.shape, proxy_fms_r.flatten(1).shape())
                #     for b in range(bs):
                #         lgr += gfk.fit(proxy_fms_r[b].detach().flatten(1).T, fms_r[b].flatten(1).T)
                #         lgf += gfk.fit(proxy_fms_u[b].detach().flatten(1).T, fms_u[b].flatten(1).T)

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                loss = forget_loss + alpha * remain_loss
                # if args.GFK:
                #     loss += args.gamma * (args.lam1 * lgr / bs   - args.lam2 * lgf / bs)
                loss.backward()
                losses.append(loss.item() / batch_size)

                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in parameters:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )
                            print(n)

                optimizer.step()

                step += 1
                # if (step+1) % 20 == 0:
                #     # if args.GFK:
                #     #     print(f"step: {i}, loss: {loss:.4f}, forget_loss: {forget_loss:.4f}, remain_loss: {remain_loss:.4f}, lgr: {lgr / bs:.4f}, lgf: {lgf / bs:.4f}")
                #     # else:
                #     print(f"step: {i}, loss: {loss:.4f}, forget_loss: {forget_loss:.4f}, remain_loss: {remain_loss:.4f}")

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)

            # model.eval()
            # save_model(model, name, epoch, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
            # save_history(losses, name, classes)

    print("Time:", time.time() - start_time)
    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )

    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    # parser.add_argument(
    #     "--GFK",
    #     default=False,
    #     action='store_true',
    # )
    # parser.add_argument(
    #     "--lam1",
    #     type=float,
    #     default=0.1,
    # )
    # parser.add_argument(
    #     "--lam2",
    #     type=float,
    #     default=0.1,
    # )
    # parser.add_argument(
    #     "--gamma",
    #     type=float,
    #     default=10,
    # )
    # parser.add_argument(
    #     "--dim",
    #     type=int,
    #     default=200,
    # )
    args = parser.parse_args()

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    certain_label(
        classes,
        train_method,
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
    )

# python random_label.py --train_method full --alpha 0.5 --lr 1e-5 --epochs 5  --class_to_forget 0 --mask_path 'mask/0/with_0.5.pt' --device '0'
