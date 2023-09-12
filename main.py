from Diffusion.TrainCondition import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "sample": "ddim",
        "noise_schedule": "linear", # "linear" or "cosine"
        "epoch": 2000,
        "batch_size": 1,
        "T": 2000,
        "image_channels": 1,
        "out_channels": 1,
        "model_channels": 128,
        "channel_mult": (1, 2, 2, 2, 4),
        "num_res_blocks": 4,
        "dropout": 0.15,
        "lr": 1e-5,
        "multiplier": 2.5,
        "beta_1": 5e-5,
        "beta_T": 0.05,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./Model_Weights/",
        "training_load_weight": "ema_ckpt_1000_.pt",
        "test_load_weight": "ema_ckpt_US2Xray_1800_.pt",
        "sampled_dir": "./Sampled_Images",
        "nrow": 1,
        "supervision": "US2Xray",
        "Xray_root_train": "/home/zyh/home/zyh/data/Xray/Paired_Xray_Us/Xray_Cor/Crop/train",
        "Us_root_train": "/home/zyh/home/zyh/data/Xray/Paired_Xray_Us/Xray_Cor/US/train",
        "Xray_root_evaluation": "/home/zyh/home/zyh/data/Xray/Paired_Xray_Us/Xray_Cor/Crop/train",
        "Us_root_evaluation": "/home/zyh/home/zyh/data/Xray/Paired_Xray_Us/Xray_Cor/US/train",
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
