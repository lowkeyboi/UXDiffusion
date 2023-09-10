
import os
from typing import Dict
import numpy as np
import copy
import random

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image

from Diffusion.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Scheduler import GradualWarmupScheduler
from Diffusion.Unet_Fusion_UCA_Seg import UNetModel

from dataset import ImageDataset
from Diffusion.nn import update_ema

def train(modelConfig: Dict):

    seed = 888
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(modelConfig["device"])

    dataset = ImageDataset(
        modelConfig["Xray_root_train"],
        modelConfig["Us_root_train"],
    )
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )

    # model setup
    # net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
    #                  num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    net_model = UNetModel(
        in_channels=modelConfig["image_channels"],
        model_channels=modelConfig["model_channels"],
        out_channels=modelConfig["out_channels"],
        num_res_blocks=modelConfig["num_res_blocks"],
        attention_resolutions=tuple([16, 16]),
        dropout=0.1,
        channel_mult=modelConfig["channel_mult"],
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=1,
        use_scale_shift_norm=True
    )



    #DDPM 初始化
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["noise_schedule"]).to(device)

    ema_model_params = [copy.deepcopy(list(net_model.parameters()))]

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            mse_loss = 0.0
            for i, (images, labels) in enumerate(tqdmDataLoader):
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                # if np.random.rand() < 0.1:
                #     labels = torch.zeros_like(labels).to(device)
                # loss = trainer(x_0, labels).sum() / b ** 2.

                loss = trainer(x_0, labels).sum() / b

                # ema for DDPM and controlnet
                for rate, params in zip([0.9999], ema_model_params):
                    update_ema(params, list(net_model.parameters()), rate=rate)

                mse_loss += loss.item() / (i+1)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(
                #     net_model.parameters(), modelConfig["grad_clip"])

                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": mse_loss,
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e % 50 == 0:
            # save checkpoint
            state_dict = net_model.state_dict()
            for i, (name, _value) in enumerate(net_model.named_parameters()):
                assert name in state_dict
                state_dict[name] = ema_model_params[0][i]
            torch.save(state_dict, os.path.join(
                modelConfig["save_dir"], f'ema_ckpt_{modelConfig["supervision"]}_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    dataset = ImageDataset(
        modelConfig["Xray_root_evaluation"],
        modelConfig["Us_root_evaluation"],
        evaluation=True
    )
    batch_size = modelConfig["nrow"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True
    )

    # load model and evaluate
    with torch.no_grad():
        model = UNetModel(
            in_channels=modelConfig["image_channels"],
            model_channels=modelConfig["model_channels"],
            out_channels=modelConfig["out_channels"],
            num_res_blocks=modelConfig["num_res_blocks"],
            attention_resolutions=tuple([16, 16]),
            dropout=0.1,
            channel_mult=modelConfig["channel_mult"],
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=1,
            use_scale_shift_norm=True
        )
        ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"], sample=modelConfig["sample"]).to(device)

        id = sorted(os.listdir(modelConfig["Us_root_evaluation"]))
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i, (img, labels) in enumerate(tqdmDataLoader):
                labels = labels.to(device)

                # 保存分割图像
                # grid = make_grid(labels, nrow=modelConfig["nrow"])
                # ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                # im = Image.fromarray(ndarr).convert("L")
                # im.save(os.path.join(
                #     modelConfig["sampled_dir"],  str(i)+"_T1000_"+"Seg.png"))
                # # 保存UCA图像
                # grid = make_grid(labels_uca, nrow=modelConfig["nrow"])
                # ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                # im = Image.fromarray(ndarr).convert("L")
                # im.save(os.path.join(
                #     modelConfig["sampled_dir"], str(i) + "_T1000_" + "UCA.png"))

                # 采样
                noisyImage = torch.randn(size=[batch_size, 1, 512, 256], device=device)
                sampledImgs = sampler(noisyImage, labels)
                sampledImgs = torch.cat((labels.cpu(), img.cpu(), sampledImgs.cpu()), -1)
                grid = make_grid(sampledImgs, nrow=modelConfig["nrow"])
                ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                id_list = list(id[i])
                id_list.insert(9, "_enhanced")
                im.save(os.path.join(modelConfig["sampled_dir"], ''.join(id_list)))
