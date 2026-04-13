import argparse
import copy
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from undersample import cartesian_mask
from denoiser import Denoiser
from model_jit import JiT
from psnr import psnr, ssim
from resnet import DnCn_resnet
import util.misc as misc


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def from_torch_real_to_torch_complex(x):
    """Convert real tensor [B,2,H,W] to complex tensor [B,1,H,W]"""
    x = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    x = torch.unsqueeze(x, dim=1).to(torch.complex64)
    return x


def from_torch_complex_to_torch_real(x):
    """Convert complex tensor [B,1,H,W] to real tensor [B,2,H,W]"""
    real = torch.real(x)
    imag = torch.imag(x)
    data = torch.cat((real, imag), dim=1).to(torch.float32)
    return data


def from_torch_real_to_np_complex(x):
    """Convert real tensor to numpy complex array"""
    x = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    x = torch.unsqueeze(x, dim=1).to(torch.complex64)
    x = np.array(x.cpu())
    return x


def ema_copy_for_test(model_diff):
    """Temporarily replace model weights with EMA version for testing, return original state"""
    model_state_dict = copy.deepcopy(model_diff.state_dict())
    ema_state_dict = copy.deepcopy(model_diff.state_dict())
    for j, (name, _) in enumerate(model_diff.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_diff.ema_params1[j]
    model_diff.load_state_dict(ema_state_dict)
    print("Switched to EMA for evaluation")
    return model_state_dict


def undersample(x, shape, acc, b, sample_n=10, centred=False, device="cpu", mask=None):
    """Undersample k-space data and return zero-filled reconstruction"""
    mask_list = []
    if mask is not None:
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32).to(device)
        x_complex = from_torch_real_to_torch_complex(x)
        undersample_data = torch.fft.ifft2(torch.fft.fft2(x_complex) * mask)
        undersample_data = from_torch_complex_to_torch_real(undersample_data)
    else:
        if b != 1:
            for _ in range(b):
                m = cartesian_mask(shape, acc, sample_n=sample_n, centred=centred)
                mask_list.append(np.expand_dims(m, axis=0))
            mask = np.stack(mask_list, axis=0)
        else:
            m = cartesian_mask(shape, acc, sample_n=sample_n, centred=centred)
            mask = np.expand_dims(np.expand_dims(m, axis=0), axis=0)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32).to(device)
        x_complex = from_torch_real_to_torch_complex(x)
        undersample_data = torch.fft.ifft2(torch.fft.fft2(x_complex) * mask)
        undersample_data = from_torch_complex_to_torch_real(undersample_data)
    return undersample_data, mask


def DC_org(x, data, mask):
    """Data consistency layer"""
    x_complex = from_torch_real_to_torch_complex(x)
    data_complex = from_torch_real_to_torch_complex(data)
    x_kspace = torch.fft.fft2(x_complex)
    data_kspace = torch.fft.fft2(data_complex)
    x_dc = torch.fft.ifft2(x_kspace * (1 - mask) + data_kspace * mask)
    x_dc = from_torch_complex_to_torch_real(x_dc)
    return x_dc


def model_init(args):
    """Initialize models and parameter groups"""
    model_jit = JiT(
        depth=args.jit_depth,
        hidden_size=args.jit_hidden_size,
        num_heads=args.jit_num_heads,
        bottleneck_dim=args.jit_bottleneck_dim,
        patch_size=args.jit_patch_size,
        input_size=args.img_size,
        in_channels=args.channel,
    ).to(args.device)

    model_diff = Denoiser(model_jit, steps=args.timestep, loss=args.loss)
    model_resnet = DnCn_resnet(
        input_dim=args.channel - 2,
        num=args.num_resnet,
        dc=True,
        inner_dim=args.inner_dim_resnet,
    ).to(args.device)

    model_diff.ema_params1 = copy.deepcopy(list(model_diff.parameters()))
    model_diff.ema_params2 = copy.deepcopy(list(model_diff.parameters()))

    param_groups_diff = misc.add_weight_decay(model_diff, args.weight_decay)
    param_groups_resnet = misc.add_weight_decay(model_resnet, args.weight_decay)

    return param_groups_diff, param_groups_resnet, model_diff, model_resnet


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU device not found.")

    # Create directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)

    writer = SummaryWriter(log_dir="logs")
    device = torch.device(args.device)

    # Data loaders
    train_dataset = CustomDataset()
    val_dataset = CustomDataset()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    # Effective batch size and learning rate scaling
    eff_batch_size = args.batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    # Initialize models
    param_groups_diff, param_groups_resnet, model_diff, model_resnet = model_init(args)
    optimizer = torch.optim.AdamW(
        param_groups_diff + param_groups_resnet, lr=args.lr, betas=(0.9, 0.95)
    )

    train_step = 0
    best_val_loss = float("inf")


    fix_mask = cartesian_mask((args.img_size, args.img_size),
                args.acc,
                args.sample_acs,
                True)

    for epoch in range(args.epochs):
        # Training phase
        model_resnet.train()
        model_diff.train()
        train_loss_epoch = 0.0
        num_batches = len(train_loader)
        lr = adjust_learning_rate(optimizer, epoch, args)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):

            con_img = batch_data["con_img"].to(device)
            target_img = batch_data["target_img"].to(device)
            b, c, h, w = target_img.shape

            # Undersample
            us_data, mask = undersample(
                target_img,
                (args.img_size, args.img_size),
                args.acc,
                b,
                args.sample_acs,
                device=device,
                mask=fix_mask,
            )

            # Forward pass
            predict_mri = model_resnet(us_data, con_img, mask)
            predict_mri = DC_org(predict_mri, us_data, mask)
            residual = target_img - predict_mri
            loss = model_diff(residual, torch.cat([con_img, us_data], dim=1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model_diff.update_ema()

            train_loss_epoch += loss.item()
            train_step += 1

            # Log training loss every 100 steps
            if train_step % 100 == 0:
                avg_loss = train_loss_epoch / min(batch_idx + 1, 100)
                writer.add_scalar("Loss/train_step", avg_loss, train_step)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}", "lr": f"{lr:.2e}"})
                # Reset accumulator for next 100 steps
                train_loss_epoch = 0.0


        # Validation every 5 epochs
        if (epoch + 1) % 25 == 0:
            model_resnet.eval()
            model_diff.eval()
            val_loss_total = 0.0
            val_psnr_total = 0.0
            val_ssim_total = 0.0
            num_val_batches = 0

            with torch.no_grad():
                # Switch to EMA weights for evaluation
                original_state = ema_copy_for_test(model_diff)

                for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    con_img = batch_data["con_img"].to(device)
                    target_img = batch_data["target_img"].to(device)
                    b, c, h, w = target_img.shape

                    us_data, mask = undersample(
                        target_img,
                        (args.img_size, args.img_size),
                        args.acc,
                        b,
                        args.sample_acs,
                        centred=True,
                        device=device,
                        mask=fix_mask,
                    )

                    # Initial prediction from ResNet
                    pred_mri = model_resnet(us_data, con_img, mask)
                    pred_mri = DC_org(pred_mri, us_data, mask)

                    # Refine with diffusion model
                    residual = model_diff.generate(torch.cat([con_img, us_data], dim=1))
                    pred_mri = pred_mri + residual
                    pred_mri = DC_org(pred_mri, us_data, mask)

                    # Compute L1 loss
                    loss_val = F.l1_loss(pred_mri, target_img)
                    val_loss_total += loss_val.item()

                    # Compute PSNR/SSIM (convert to numpy)
                    pred_np = pred_mri.cpu().numpy()
                    target_np = target_img.cpu().numpy()

                    for i in range(b):
                        # Use magnitude image (sqrt of sum of squares of real/imag channels)
                        pred_mag = np.sqrt(pred_np[i, 0] ** 2 + pred_np[i, 1] ** 2)
                        target_mag = np.sqrt(target_np[i, 0] ** 2 + target_np[i, 1] ** 2)
                        val_psnr_total += psnr(pred_mag, target_mag, data_range=target_mag.max())
                        val_ssim_total += ssim(pred_mag, target_mag, data_range=target_mag.max())

                    num_val_batches += 1

                # Restore original weights
                model_diff.load_state_dict(original_state)

            avg_val_loss = val_loss_total / (num_val_batches * b) if num_val_batches > 0 else 0.0
            avg_val_psnr = val_psnr_total / (num_val_batches * b) if num_val_batches > 0 else 0.0
            avg_val_ssim = val_ssim_total / (num_val_batches * b) if num_val_batches > 0 else 0.0

            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Metrics/PSNR", avg_val_psnr, epoch)
            writer.add_scalar("Metrics/SSIM", avg_val_ssim, epoch)

            print(f"Validation Loss: {avg_val_loss:.6f}, PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join("weights", f"best_model_epoch{epoch+1}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_resnet_state_dict": model_resnet.state_dict(),
                        "model_diff_state_dict": model_diff.state_dict(),
                        "ema_params1": model_diff.ema_params1,
                        "ema_params2": model_diff.ema_params2,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": args,
                    },
                    save_path,
                )
                print(f"Saved best model to {save_path} (val loss: {best_val_loss:.6f})")

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Reconstruction with JIT and Diffusion")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--acc", type=int, default=4, help="Acceleration factor")
    parser.add_argument("--sample_acs", type=int, default=10, help="Number of ACS lines")
    parser.add_argument("--loss", type=str, default="l1", help="Loss type: l1 or l2")
    parser.add_argument("--channel", type=int, default=6, help="Input channels for JIT")
    parser.add_argument("--num_resnet", type=int, default=8, help="Number of ResNet blocks")
    parser.add_argument("--inner_dim_resnet", type=int, default=32, help="Inner dimension of ResNet")
    parser.add_argument("--jit_depth", type=int, default=6, help="Depth of JIT")
    parser.add_argument("--jit_hidden_size", type=int, default=384, help="Hidden size of JIT")
    parser.add_argument("--jit_num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--jit_bottleneck_dim", type=int, default=64, help="Bottleneck dimension of JIT")
    parser.add_argument("--jit_patch_size", type=int, default=6, help="Patch size for JIT")
    parser.add_argument("--timestep", type=int, default=1, help="Diffusion timesteps")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="LR schedule: constant or cosine")
    parser.add_argument("--lr", type=float, default=None, help="Absolute learning rate")
    parser.add_argument("--blr", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum LR for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    args = parser.parse_args()
    main(args)