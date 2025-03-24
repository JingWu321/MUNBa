import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_nsfw_data,
    setup_model,
)
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm

import gc
from timm.utils import AverageMeter
from timm.models.layers import trunc_normal_

import copy
import timm
import math
import time
import random

import cvxpy as cp


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


word_wear = "a photo of a person wearing clothes"


def MUNBa(classes,
        train_method,
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
        with_l1,
        beta,
        alpha,
):
    # MODEL TRAINING SETUP
    # print(config_path)
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size, forget_path='./dataFolder/NSFW',
        remain_path='./dataFolder/NotNSFW', image_size=image_size)

    #### Convex Optimization Problem (bargaining game) Initialization ####
    if args.nash:
        n_tasks = 2 # K
        init_gtg = np.eye(n_tasks) # G^T G: gradient matrix product, shape: [K, K]
        G_param = cp.Parameter(shape=(n_tasks, n_tasks), value=init_gtg) # will be updated in-loop with the current GTG
        normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0])) # will be updated in-loop with torch.norm(GTG).detach().cpu().numpy().reshape((1,))
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
        prvs_alpha[0] = 1.0
        prvs_alpha[1] = 0.5
    #####################################################


    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                print(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    print(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    print(name)
                    parameters.append(param)

    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-MUNBa-mask-method_{train_method}-lr_{lr}_E{epochs}"
    else:
        name = f"compvis-nsfw-MUNBa-method_{train_method}-lr_{lr}_E{epochs}"

    # NSFW Removal
    word_wear = "a photo of a person wearing clothes"
    word_print = 'nsfw'.replace(" ", "")

    # TRAINING CODE
    step = 0
    start_time = time.time()
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time_1:
            for i, batch in enumerate(forget_dl):
                model.train()
                optimizer.zero_grad()
                forget_batch = next(iter(forget_dl))
                remain_batch = next(iter(remain_dl))

                loss_r = model.shared_step(remain_batch)[0]

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_prompts = [word_wear] * forget_batch['jpg'].size(0)
                pseudo_batch = {
                    "jpg": forget_batch['jpg'],
                    "txt": pseudo_prompts,
                }
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=model.device,).long()
                noise = torch.randn_like(forget_input, device=model.device)
                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                loss_u = criteria(forget_out, pseudo_out) * beta
                # loss_u = -model.shared_step(forget_batch)[0]

                #####################################################
                if args.munba:
                    # compute gradient for each task
                    grads = {}
                    for task, loss in zip([0, 1], [loss_r, loss_u]):
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(loss, parameters, retain_graph=True)[0]
                        grads[task] = torch.cat([torch.flatten(g.detach()) for g in grad])

                    ############# [1] Choose to use the iterative solution
                    prvs_alpha = return_weights(grads, prvs_alpha, G_param, normalization_factor_param,
                                                alpha_param, prvs_alpha_param, prob)
                    print(f'prvs_alpha: {prvs_alpha}')
                    if np.all(prvs_alpha == 1): # Bargaining failed
                        # continue
                        loss = loss_r + loss_u * 0.1
                    else:
                        loss = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]

                    # ############ [2] Choose to use the closed-form solution
                    # g1 = torch.dot(grads[0], grads[0])
                    # g2 = torch.dot(grads[0], grads[1])
                    # g3 = torch.dot(grads[1], grads[1])
                    # prvs_alpha[0] = torch.sqrt( (g1*g3 - g2*torch.sqrt(g1*g3)) / (g1*g1*g3 - g1*g2*g2 + 1e-8) )
                    # prvs_alpha[1] = (1 - g1 * prvs_alpha[0] * prvs_alpha[0]) / (g2*prvs_alpha[0] + 1e-8)
                    # print(f'prvs_alpha: {prvs_alpha}')
                    # if prvs_alpha[0] > 0 and prvs_alpha[1] > 0: # Bargaining succeeded
                    #     loss = loss_r * prvs_alpha[0] + loss_u * prvs_alpha[1]
                    # else:
                    #     # continue
                    #     loss = loss_r + loss_u * 0.5
                #####################################################
                else:
                    loss = loss_r + args.lam * loss_u

                optimizer.zero_grad()
                if with_l1:
                    loss = loss + alpha * l1_regularization(parameters)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                losses.append(loss.item() / batch_size)

                optimizer.step()
                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if (step+1) % 10 == 0:
                    if args.nash:
                        print(f"step: {step}, loss: {loss:.4f}, loss_r: {loss_r * prvs_alpha[0]:.4f}, loss_u: {loss_u * prvs_alpha[1]:.4f}")
                    else:
                        print(f"step: {step}, loss: {loss:.4f}, loss_r: {loss_r:.4f}, loss_u: {args.lam * loss_u:.4f}")
                    save_history(losses, name, word_print)

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)

            model.eval()
            save_model(model, name, epoch, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

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
    save_history(losses, name, word_print)


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
            name, compvis_config_file, diffusers_config_file, device=device, num=num,
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)



def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
        default="configs/stable-diffusion/v1-inference-nash.yaml",
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
        default="4",
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
    parser.add_argument("--with_l1", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1e-4)
    ##################################### MUNBa setting #################################################
    parser.add_argument("--munba", default=False, action='store_true',)
    parser.add_argument("--beta", type=float, default=100.0)

    args = parser.parse_args()

    setup_seed(42)

    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
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

    MUNBa(
        classes,
        train_method,
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
        args.with_l1,
        args.beta,
        args.alpha,
    )



