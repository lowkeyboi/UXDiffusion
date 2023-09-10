import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
from tqdm.auto import tqdm

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, noise_schedule="linear"):
        super().__init__()

        self.model = model
        self.T = T

        schedule_name = noise_schedule
        if schedule_name == "linear":
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        elif schedule_name == "cosine":
            self.register_buffer('betas', torch.tensor(betas_for_alpha_bar(T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)).double())
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0., sample="ddpm", noise_schedule="linear"):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        schedule_name = noise_schedule
        if schedule_name == "linear":
            self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        elif schedule_name == "cosine":
            self.register_buffer('betas', torch.tensor(
                betas_for_alpha_bar(T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, )).double())
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('alphas_bar', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_bar_prev', F.pad(alphas_bar, [1, 0], value=1)[:T])
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        # DDPM 使用
        self.register_buffer('sqrt_recip_alphas_cumprod', np.sqrt(1.0 / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', np.sqrt(1.0 / alphas_bar - 1))
        self.register_buffer('posterior_mean_coef1', (self.betas * np.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)))
        self.register_buffer('posterior_mean_coef2', ((1.0 - alphas_bar_prev)
                * np.sqrt(alphas)
                / (1.0 - alphas_bar)))

        # DDIM 使用 DDIM 中 alphas = DDPM中的 alphas_bar
        self.sigma = 0.0
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas_bar))
        # direction
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)

        self.sample = sample


    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels, sample="ddpm"):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        # nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        # eps = (1. + self.w) * eps - self.w * nonEps
        # xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        if sample=="ddpm":
            #先算 pred_x0
            pred_x0 = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            pred_x0 = pred_x0.clamp(-1, 1)
            #通过pred_x0再算后验均值
            xt_prev_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            return xt_prev_mean, var
        else:
            # ddim pred_x0
            pred_x0 = (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)*eps) / extract(self.sqrt_alphas, t, x_t.shape)
            # pred_x0 = pred_x0.clamp(-1, 1)
            direction_xt = torch.sqrt(1.0 - extract(self.alphas_bar_prev, t, x_t.shape)) * eps
            x_prev = torch.sqrt(extract(self.alphas_bar_prev, t, x_t.shape)) * pred_x0 + direction_xt
            return x_prev, pred_x0

    # def ddim_sampling_parameters(self):
    #     return

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        torch.manual_seed(666)

        # x_T_interpolation = torch.zeros(size=[1, 1, 256, 128]).to(x_T.device)
        # import torch.nn.functional as F
        # x_T = F.interpolate(x_T_interpolation, scale_factor=2)

        # def slerp(z1, z2, alpha):
        #     theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        #     return (
        #         torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        #         + torch.sin(alpha * theta) / torch.sin(theta) * z2
        #     ) 
        # x_T = slerp(x_T, x_T_interpolation, 0.5)

        x_t = x_T
        if self.sample == "ddpm":
            with tqdm(reversed(range(self.T)), dynamic_ncols=True, total=self.T) as reverse_process:
                for time_step in reverse_process:
                    t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                    mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
                    if time_step > 0:
                        noise = torch.randn_like(x_t)
                    else:
                        noise = 0
                    # x_t = mean + torch.exp(0.5 * log_var) * noise
                    x_t = mean + torch.sqrt(var) * noise
                    assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        elif self.sample == "ddim":
            ddim_steps = 50
            c = 2000 // ddim_steps
            ddim_timesteps = np.asarray(list(range(0, 2000, c)))
            ddim_timestep_seq = ddim_timesteps + 1
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
            for time_step in reversed(range(0, ddim_steps)):
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * ddim_timestep_seq[time_step]
                prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * ddim_timestep_prev_seq[time_step]

                alpha_cumprod_t = extract(self.alphas_bar, t, x_t.shape)
                alpha_cumprod_t_prev = extract(self.alphas_bar, prev_t, x_t.shape)

                pred_noise = self.model(x_t, t, labels)

                pred_x0 = (x_t - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, min=0.0, max=1.0)

                ddim_eta = 0.0
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x_t)
                # x_prev = torch.clamp(x_prev, min=-1., max=1.)
                x_t = x_prev


                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t

        return torch.clip(x_0, 0, 1)


