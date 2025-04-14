import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def SoftIoULoss(pred, target):
    # Old One
    pred = torch.sigmoid(pred)
    smooth = 1

    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)

    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

    # print(intersection.sum() + smooth)
    # print(pred.sum() + target.sum() - intersection.sum() + smooth)

    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)

    loss = 1 - loss.mean()
    # loss = loss.mean()
    # loss = (1 - loss).mean()


    # return loss, (intersection.sum() + smooth), (pred.sum() + target.sum() - intersection.sum() + smooth), pred.sum(), pred
    return loss


# def binary_cross_entropy(y_true, y_pred):
#     # 对预测值应用 sigmoid 激活
#     # y_pred = torch.sigmoid(y_pred)
#
#     # 计算交叉熵损失
#     loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
#
#     return loss.mean()  # 返回平均损失


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_reverse = alphas[::-1]
        alphas_reverse_cumprod = np.cumprod(alphas_reverse, axis=0)
        self.register_buffer('alphas_reverse_cumprod', to_torch(alphas_reverse_cumprod))

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior_cross_steps(self, x_start, x_t, t, n):
        posterior_mean_coef1 = self.sqrt_alphas_cumprod_prev[t-n] * (1-self.alphas_reverse_cumprod[n-1]) / (1-self.alphas_cumprod[t])

        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, x_label=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            if self.denoise_fn.deep_supervision:
                # noises, _ = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
                noises = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=noises[-1])
                # x_recon = noises[-1]
            else:
                # noise, _ = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
                noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=noise)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        loss = SoftIoULoss(x_recon, x_label)

        if clip_denoised:
            # x_recon.clamp_(-1., 1.)
            # 创建一个全零的张量
            result = torch.full_like(x_recon, fill_value=-1)

            # 将大于 0 的位置设为 1
            result[x_recon > 0] = 1
            x_recon = result

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, loss, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, x_label=None):
        model_mean, model_log_variance, loss, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, x_label=x_label)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), loss, x_recon

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, x_label=None):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        x_recon = None
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn((shape[0],1,shape[2],shape[3]), device=device)
            ret_img = img
            ret_loss = {}
            x0_img = None
            x0_timestamps = []
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img, iou_loss, x_recon = self.p_sample(img, i, condition_x=x, x_label=x_label)

                if i == (self.num_timesteps-1) or i % sample_inter == 0:
                # if i <= 10:
                #     ret_img = torch.cat([ret_img, img], dim=1)
                #     ret_loss[i] = iou_loss
                    if x0_img is not None:
                       
                        x0_img = torch.cat([x0_img, x_recon], dim=1)
                        x0_timestamps.append(i)
                    else:
                   
                        x0_img = x_recon
                        x0_timestamps.append(i)

                    # if i == 1:
                    # for row in pred[0][0].cpu():
                    #     print(" ".join(f"{x.item():.10f}" for x in row))
                    # exit(0)

                if i == 0:
                    ret_img = img
                #     print(fenzi)
                #     print(fenmu)
                #     print(pred_sum)
                #     for row in pred[0][0].cpu():
                #         print(" ".join(f"{x.item():.2f}" for x in row))
                #     exit(0)
        if continous:
            # return ret_img, ret_loss, x_recon
        
            return x0_img, x0_timestamps
        else:
            return ret_img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous, )

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, x_label=None):
        return self.p_sample_loop(x_in, continous, x_label)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # print(x_start.shape)
        # print(continuous_sqrt_alpha_cumprod.shape)
        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, iter, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                # self.sqrt_alphas_cumprod_prev[t],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        # if t <= 150:

        # if random.random() < 0.3:
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise)
        # else:
        #     x_noisy = self.q_sample(
        #         x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
        #         noise=noise)
        # else:
        #     x_noisy = self.q_sample(
        #         x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # x_recon, preds = self.denoise_fn(
            #     torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            if (iter + 1) % 10 == 0:
                # in_ch =1，只有噪声
                # 提取与剩余通道（假设是第0通道）对应的权重
                new_weight = self.denoise_fn.conv_first.weight[:, 3:4, :, :].clone()  # 只保留第0通道的权重
                new_bias = self.denoise_fn.conv_first.bias.clone()  # 偏置不变

                # 将提取的权重和偏置赋值给新的卷积层
                self.denoise_fn.conv_first_c1.weight = nn.Parameter(new_weight)
                self.denoise_fn.conv_first_c1.bias = nn.Parameter(new_bias)
                x_recon = self.denoise_fn(
                    x_noisy, continuous_sqrt_alpha_cumprod)
            else:
                # in_ch= 4,输入为[x_con,noise]
                # 提取与剩余通道（假设是第0通道）对应的权重
                new_weight = self.denoise_fn.conv_first_c1.weight[:, 0:1, :, :].clone()  # 只保留第0通道的权重
                new_bias = self.denoise_fn.conv_first_c1.bias.clone()  # 偏置不变

                origin_weight = self.denoise_fn.conv_first.weight[:, 0:3, :, :].clone()

                full_weight = torch.cat([origin_weight, new_weight], dim=1)

                # 将提取的权重和偏置赋值给新的卷积层
                self.denoise_fn.conv_first.weight = nn.Parameter(full_weight)
                self.denoise_fn.conv_first.bias = nn.Parameter(new_bias)
                x_recon = self.denoise_fn(
                    torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)

        ### conditional network training loss ###
        ######

        # if iter % 10 == 0:
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #
        #     anc_loss = 0
        #     for pred in preds:
        #         anc_loss += SoftIoULoss(pred, x_in['MK'])
        #     anc_loss /= len(preds)
        #
        #     return loss, st_loss, anc_loss
        # else:
        #
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (
        #                         1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (
        #                     1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #     return loss, st_loss, loss


        ###

        criterion = nn.BCEWithLogitsLoss()

        if self.denoise_fn.deep_supervision:
            loss = 0
            st_loss = 0
            bse_loss = 0
            for x_re in x_recon:
                x_st = (x_noisy - (
                        1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod

                # x_st = x_re
                st_loss += SoftIoULoss(x_st, x_in['MK'])
                # loss += self.loss_func(noise, x_re)
                # x_im = x_st.clip(-1, 1)
                # x_im = (x_im + 1) / 2
                bse_loss += criterion(x_st, x_in['MK'])
            # loss /= len(x_recon)
            st_loss /= len(x_recon)
            bse_loss /= len(x_recon)
            loss = st_loss + bse_loss

        else:
            # loss = self.loss_func(noise, x_recon)
            x_st = (x_noisy - (
                    1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod

            st_loss = SoftIoULoss(x_st, x_in['MK'])

            # x_im = x_st.clip(-1, 1)
            # x_im = (x_im + 1) / 2
            bse_loss = criterion(x_st, x_in['MK'])

            loss = st_loss + bse_loss


        return loss, st_loss, bse_loss

    def p_losses_correct(self, x_in, iter, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)

        x_st_last = x_start

        ## t+1
        if iter > 6e4 and t < self.num_timesteps and random.random() < 0.4:
            last_t = random.randint(t+1, self.num_timesteps + 1)
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[last_t-1],
                    # self.sqrt_alphas_cumprod_prev[t],
                    self.sqrt_alphas_cumprod_prev[last_t],
                    size=b
                )
            ).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)

            noise = default(noise, lambda: torch.randn_like(x_start))

            x_noisy = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

            if not self.conditional:
                with torch.no_grad():
                    x_recon_last = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            else:
                # x_recon, preds = self.denoise_fn(
                #     torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
                with torch.no_grad():
                    x_recon_last = self.denoise_fn(
                        torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)

            x_st_last = (x_noisy - (
                    1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon_last[-1]) / continuous_sqrt_alpha_cumprod

        ### conditional network training loss ###
        ######

        # if iter % 10 == 0:
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #
        #     anc_loss = 0
        #     for pred in preds:
        #         anc_loss += SoftIoULoss(pred, x_in['MK'])
        #     anc_loss /= len(preds)
        #
        #     return loss, st_loss, anc_loss
        # else:
        #
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (
        #                         1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (
        #                     1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #     return loss, st_loss, loss


        ###

            del x_noisy
            # del noise
            torch.cuda.empty_cache()

        continuous_sqrt_alpha_cumprod_curr = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                # self.sqrt_alphas_cumprod_prev[t],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod_curr = continuous_sqrt_alpha_cumprod_curr.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(
            x_start=x_st_last, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod_curr.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod_curr)
        else:
            # x_recon, preds = self.denoise_fn(
            #     torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod_curr)

        continuous_sqrt_alpha_cumprod_curr = continuous_sqrt_alpha_cumprod_curr.view(-1, 1, 1, 1)

        criterion = nn.BCEWithLogitsLoss()

        if self.denoise_fn.deep_supervision:
            loss = 0
            st_loss = 0
            bse_loss = 0
            for x_re in x_recon:
                x_st = (x_noisy - (
                        1 - continuous_sqrt_alpha_cumprod_curr ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod_curr

                st_loss += SoftIoULoss(x_st, x_in['MK'])
                loss += self.loss_func(noise, x_re)
                # x_im = x_st.clip(-1, 1)
                # x_im = (x_im + 1) / 2
                bse_loss += criterion(x_st, x_in['MK'])
            loss /= len(x_recon)
            st_loss /= len(x_recon)
            bse_loss /= len(x_recon)
        else:
            loss = self.loss_func(noise, x_recon)
            x_st = (x_noisy - (
                    1 - continuous_sqrt_alpha_cumprod_curr ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod_curr

            st_loss = SoftIoULoss(x_st, x_in['MK'])

            # x_im = x_st.clip(-1, 1)
            # x_im = (x_im + 1) / 2
            bse_loss = criterion(x_st, x_in['MK'])

        return loss, st_loss, bse_loss

    def p_losses_correct_last_infer(self, x_in, iter, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)

        x_noisy = None

        ## t+1
        if iter > 6e4 and t < self.num_timesteps and random.random() < 0.4:
            last_t = t + 1
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[last_t-1],
                    # self.sqrt_alphas_cumprod_prev[t],
                    self.sqrt_alphas_cumprod_prev[last_t],
                    size=b
                )
            ).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)

            noise = default(noise, lambda: torch.randn_like(x_start))

            x_noisy_last = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

            with torch.no_grad():
                x_noisy, _, _ = self.p_sample(x_noisy_last, last_t, condition_x=x_in['SR'], x_label=x_in['MK'])

        ### conditional network training loss ###
        ######

        # if iter % 10 == 0:
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #
        #     anc_loss = 0
        #     for pred in preds:
        #         anc_loss += SoftIoULoss(pred, x_in['MK'])
        #     anc_loss /= len(preds)
        #
        #     return loss, st_loss, anc_loss
        # else:
        #
        #     if self.denoise_fn.deep_supervision:
        #         loss = 0
        #         st_loss = 0
        #         for x_re in x_recon:
        #             x_st = (x_noisy - (
        #                         1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod
        #
        #
        #             st_loss += SoftIoULoss(x_st, x_in['MK'])
        #             loss += self.loss_func(noise, x_re)
        #         loss /= len(x_recon)
        #         st_loss /= len(x_recon)
        #     else:
        #         loss = self.loss_func(noise, x_recon)
        #         x_st = (x_noisy - (
        #                     1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod
        #
        #         st_loss = SoftIoULoss(x_st, x_in['MK'])
        #     return loss, st_loss, loss


        ###

            # del x_noisy
            # del noise
            torch.cuda.empty_cache()

        continuous_sqrt_alpha_cumprod_curr = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                # self.sqrt_alphas_cumprod_prev[t],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod_curr = continuous_sqrt_alpha_cumprod_curr.view(
            b, -1)

        # noise = default(noise, lambda: torch.randn_like(x_start))

        if x_noisy is None:
            x_noisy = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod_curr.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod_curr)
        else:
            # x_recon, preds = self.denoise_fn(
            #     torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod_curr)

        continuous_sqrt_alpha_cumprod_curr = continuous_sqrt_alpha_cumprod_curr.view(-1, 1, 1, 1)

        criterion = nn.BCEWithLogitsLoss()

        if self.denoise_fn.deep_supervision:
            loss = 0
            st_loss = 0
            bse_loss = 0
            for x_re in x_recon:
                x_st = (x_noisy - (
                        1 - continuous_sqrt_alpha_cumprod_curr ** 2).sqrt() * x_re) / continuous_sqrt_alpha_cumprod_curr

                st_loss += SoftIoULoss(x_st, x_in['MK'])
                loss += self.loss_func(noise, x_re)
                # x_im = x_st.clip(-1, 1)
                # x_im = (x_im + 1) / 2
                bse_loss += criterion(x_st, x_in['MK'])
            loss /= len(x_recon)
            st_loss /= len(x_recon)
            bse_loss /= len(x_recon)
        else:
            loss = self.loss_func(noise, x_recon)
            x_st = (x_noisy - (
                    1 - continuous_sqrt_alpha_cumprod_curr ** 2).sqrt() * x_recon) / continuous_sqrt_alpha_cumprod_curr

            st_loss = SoftIoULoss(x_st, x_in['MK'])

            # x_im = x_st.clip(-1, 1)
            # x_im = (x_im + 1) / 2
            bse_loss = criterion(x_st, x_in['MK'])

        return loss, st_loss, bse_loss

    def forward(self, x, iter, *args, **kwargs):
        # import random
        # if iter < 6e4:
        #     return self.p_losses(x, iter, *args, **kwargs)
        #
        # if random.random() < 0.6:
        #     return self.p_losses(x, iter, *args, **kwargs)
        # else:
        return self.p_losses(x, iter, *args, **kwargs)
