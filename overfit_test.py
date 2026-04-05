"""
Note: This test was run on 128x128 images so you need to adjust the code if you want to run it on 64x64 images or update the architecture.
"""

import torch
import os
from dataset import CelebADataset
from architecture import CelebAVAE, ELBOLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_every_steps = 1
    batch_size = 8

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/reconstructions', exist_ok=True)
    writer = SummaryWriter(log_dir='runs/beta_vae')

    dataset_path = "./dataset"
    img_dir = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CelebADataset(root_dir=img_dir, start=0, end=7, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True, prefetch_factor=1)

    train_preview_batch = next(iter(data_loader))

    model = CelebAVAE(latent_dim=128).to(device)
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = ELBOLoss(gamma=0.0, C_max=None, device=device)

    epochs = 300
    warmup_epochs = 15

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    for epoch in range(epochs):
        model.train()
        total_loss = total_kl = total_recon = total_images = 0.0
        train_kl_per_dim_sum = None

        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit='batch', leave=False)

        for step, image in enumerate(loop):
            if device == 'cuda':
                image = image.to(device, memory_format=torch.channels_last)
            else:
                image = image.to(device)
            global_step = step + epoch * len(data_loader)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=torch.bfloat16):
                x_hat, mu, logvar = model(image)
                loss, recon, kl, kl_per_dim = criterion(image, x_hat, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_kl += kl.item() * batch_size
            total_recon += recon.item() * batch_size
            total_images += batch_size
            if train_kl_per_dim_sum is None:
                train_kl_per_dim_sum = kl_per_dim.detach() * batch_size
            else:
                train_kl_per_dim_sum += kl_per_dim.detach() * batch_size

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                kl=f"{kl.item():.4f}",
                recon=f"{recon.item():.4f}",
                avg_loss=f"{total_loss/total_images:.4f}",
                active_dims=(kl_per_dim > 0.01).sum().item()
            )

            if (global_step + 1) % log_every_steps == 0:
                writer.add_scalar('train_step/loss', loss.item(), global_step + 1)
                writer.add_scalar('train_step/recon', recon.item(), global_step + 1)
                writer.add_scalar('train_step/kl', kl.item(), global_step + 1)
                writer.add_scalar('train_step/active_dims', (kl_per_dim > 0.01).sum().item(), global_step + 1)
                writer.add_scalar('train_step/lr', optimizer.param_groups[0]['lr'], global_step + 1)
            
        scheduler.step()

        train_loss = total_loss / total_images
        train_recon = total_recon / total_images
        train_kl = total_kl / total_images
        avg_train_kl_per_dim = train_kl_per_dim_sum / total_images

        writer.add_scalar('train/loss', train_loss, epoch + 1)
        writer.add_scalar('train/recon', train_recon, epoch + 1)
        writer.add_scalar('train/kl', train_kl, epoch + 1)
        for dim, dim_kl in enumerate(avg_train_kl_per_dim):
            writer.add_scalar(f'train/kl_per_dim/dim_{dim}', dim_kl.item(), epoch + 1)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            if train_preview_batch is not None:
                preview_batch = train_preview_batch.to(device, memory_format=torch.channels_last) if device == 'cuda' else train_preview_batch.to(device)

                with torch.no_grad():
                    with autocast(device_type=device, dtype=torch.bfloat16):
                        train_recon_preview, _, _ = model(preview_batch)

                original = (preview_batch * 0.5 + 0.5).clamp(0, 1).float()
                reconstructed = (train_recon_preview * 0.5 + 0.5).clamp(0, 1).float()
                comparison_grid = vutils.make_grid(torch.cat([original, reconstructed], dim=0), nrow=8)

                recon_path = f"checkpoints/reconstructions/epoch_{epoch+1:03d}.png"
                vutils.save_image(comparison_grid, recon_path)
                writer.add_image('train/reconstructions', comparison_grid, epoch + 1)

            writer.flush()

    writer.close()

if __name__ == "__main__":
    main()