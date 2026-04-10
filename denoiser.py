import torch
import torch.nn as nn


def DC(x, data, mask):
    x = from_torch_real_to_torch_complex(x)
    data = from_torch_real_to_torch_complex(data)
    x = torch.fft.ifft2(torch.fft.fft2(x) * (1 - mask) + torch.fft.fft2(data) * mask)
    x = from_torch_complex_to_torch_real(x)
    return x


def from_torch_real_to_torch_complex(x):
    x = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    x = torch.unsqueeze(x, dim=1).to(torch.complex64)
    return x


def from_torch_complex_to_torch_real(x):
    real = torch.real(x)
    imag = torch.imag(x)
    data = torch.cat((real, imag), dim=1).to(torch.float32)
    return data


class Denoiser(nn.Module):
    def __init__(
            self, net_worker, img_size=96, steps=50, loss='l1'
    ):
        super().__init__()
        self.net = net_worker
        self.img_size = img_size

        self.P_mean = -0.8
        self.P_std = 0.8
        self.t_eps = 5e-2
        self.noise_scale = 1.0

        # ema
        self.ema_decay1 = 0.9999
        self.ema_decay2 = 0.9996
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = 'heun'
        self.steps = steps
        self.loss = loss

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)


    # x 为级联 resnet 与 ground truth 的残差   con_img为所有条件的 concat 结果  loss 只需 flow matching 返回的loss 级联 Resnet 需要损失函数
    def forward(self, x, con_img):

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), con_img)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        if self.loss == 'l1':

            loss = torch.abs(v - v_pred)
            loss = loss.mean(dim=(1, 2, 3)).mean()

        else:

            # l2 loss
            loss = (v - v_pred) ** 2
            loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, con_img, mask=None, us_data=None):

        device = con_img.device
        bsz = con_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 2, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz,
                                                                                                             -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # 确保mask和us_data在正确的设备上
        if mask is not None:
            mask = mask.to(device)
        if us_data is not None:
            us_data = us_data.to(device)

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, con_img, mask=mask, us_data=us_data)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], con_img)

        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond_img, mask=None, us_data=None):
        # conditional
        x_cond = self.net(z, t.flatten(), cond_img)
        if mask is not None and us_data is not None:
            x_cond = DC(x_cond, us_data, mask)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        return v_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond_img, mask=None, us_data=None):
        v_pred = self._forward_sample(z, t, cond_img, mask=mask, us_data=us_data)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond_img, mask=None, us_data=None):
        v_pred_t = self._forward_sample(z, t, cond_img, mask=mask, us_data=us_data)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, cond_img)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
