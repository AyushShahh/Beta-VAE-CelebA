import os
import torch
from architecture import CelebAVAE, ELBOLoss
import kagglehub
from torchvision import transforms
from torchvision import utils as vutils
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from torch.amp import autocast

class CelebADataset(Dataset):
    def __init__(self, root_dir, start, end, transform=None):
        assert end >= start, "End index must be greater or equal than start index"
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(root_dir))[start:end+1]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        with Image.open(img_name) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_every_steps = 1
    batch_size = 128 # 96

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/train_reconstructions', exist_ok=True)
    os.makedirs('checkpoints/val_reconstructions', exist_ok=True)
    writer = SummaryWriter(log_dir='runs/beta_vae')

    # load dataset
    dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset", output_dir="./dataset")
    # dataset_path = "./dataset"
    img_dir = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
    
    transform = transforms.Compose([
        transforms.CenterCrop(148),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebADataset(root_dir=img_dir, start=0, end=182636, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=2)
    val_dataset = CelebADataset(root_dir=img_dir, start=182637, end=202598, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=2)

    model = CelebAVAE(latent_dim=32).to(device)
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model, mode='default', dynamic=False)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    criterion = ELBOLoss(gamma=1000.0, C_max=50.0, device=device)

    epochs = 200
    # warmup_epochs = 20
    capacity_warmup_steps = 80 * len(data_loader)
    # lpips = 0
    val_preview_batch = next(iter(val_loader))[:8].to(device)
    train_preview_batch = next(iter(data_loader))[:8].to(device)
    
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    for epoch in range(epochs):
        model.train()
        total_loss = total_kl = total_recon = total_images = 0.0
        train_kl_per_dim_sum = None
        # if epoch + 1 > warmup_epochs:
        #     lpips = 0.01
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
                loss, recon, kl, kl_per_dim, C, mu, logvar = criterion(image, x_hat, mu, logvar, global_step, capacity_warmup_steps)

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            # clip_coef = min(1.0, 1.0 / (total_norm.item() + 1e-6))
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_kl += kl.item() * batch_size
            total_recon += recon.item() * batch_size
            total_images += batch_size
            if train_kl_per_dim_sum is None:
                train_kl_per_dim_sum = kl_per_dim.detach() * batch_size
            else:
                train_kl_per_dim_sum += kl_per_dim.detach() * batch_size

            # loop.set_postfix(
            #     loss=f"{loss.item():.4f}",
            #     kl=f"{kl.item():.4f}",
            #     recon=f"{recon.item():.4f}",
            #     avg_loss=f"{total_loss/total_images:.4f}",
            #     active_dims=(kl_per_dim > 0.01).sum().item()
            # )

            if (global_step + 1) % log_every_steps == 0:
                writer.add_scalar('train_step/loss', loss.item(), global_step + 1)
                writer.add_scalar('train_step/recon', recon.item(), global_step + 1)
                writer.add_scalar('train_step/kl', kl.item(), global_step + 1)
                writer.add_scalar('train_step/active_dims', (kl_per_dim > 0.1).sum().item(), global_step + 1)
                writer.add_scalar('train_step/strong_dims', (kl_per_dim > 1.0).sum().item(), global_step + 1)
                writer.add_scalar('train_step/total_norm', total_norm.item(), global_step + 1)
                # writer.add_scalar('train_step/lr', optimizer.param_groups[0]['lr'], global_step + 1)
                writer.add_scalar('train_step/C', C, global_step + 1)
                writer.add_scalar('train_step/mu', mu.mean().item(), global_step + 1)
                writer.add_scalar('train_step/logvar', logvar.mean().item(), global_step + 1)
                # writer.add_scalar('train_step/grad_norm_scale', clip_coef, global_step + 1)
        # scheduler.step()

        train_loss = total_loss / total_images
        train_recon = total_recon / total_images
        train_kl = total_kl / total_images
        avg_train_kl_per_dim = train_kl_per_dim_sum / total_images

        writer.add_scalar('train/loss', train_loss, epoch + 1)
        writer.add_scalar('train/recon', train_recon, epoch + 1)
        writer.add_scalar('train/kl', train_kl, epoch + 1)
        for dim, dim_kl in enumerate(avg_train_kl_per_dim):
            writer.add_scalar(f'train/kl_per_dim/dim_{dim}', dim_kl.item(), epoch + 1)

        model.eval()
        val_total_loss = val_total_kl = val_total_recon = val_total_images = 0.0
        val_kl_per_dim_sum = None
        val_preview_batch = None

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}", unit='batch', leave=False)
            for val_image in val_loop:
                if device == 'cuda':
                    val_image = val_image.to(device, memory_format=torch.channels_last)
                else:
                    val_image = val_image.to(device)

                if val_preview_batch is None:
                    val_preview_batch = val_image[:8].clone()

                with autocast(device_type=device, dtype=torch.bfloat16):
                    val_x_hat, val_mu, val_logvar = model(val_image)
                    val_loss, val_recon, val_kl, val_kl_per_dim, _, _, _ = criterion(
                        val_image,
                        val_x_hat,
                        val_mu,
                        val_logvar,
                        step=global_step,
                        total_steps=capacity_warmup_steps
                    )

                val_total_loss += val_loss.item() * batch_size
                val_total_recon += val_recon.item() * batch_size
                val_total_kl += val_kl.item() * batch_size
                val_total_images += batch_size
                if val_kl_per_dim_sum is None:
                    val_kl_per_dim_sum = val_kl_per_dim.detach() * batch_size
                else:
                    val_kl_per_dim_sum += val_kl_per_dim.detach() * batch_size

                val_loop.set_postfix(
                    loss=f"{val_loss.item():.4f}",
                    kl=f"{val_kl.item():.4f}",
                    recon=f"{val_recon.item():.4f}",
                    avg_loss=f"{val_total_loss/val_total_images:.4f}"
                )

        val_loss = val_total_loss / val_total_images
        val_recon = val_total_recon / val_total_images
        val_kl = val_total_kl / val_total_images
        avg_val_kl_per_dim = val_kl_per_dim_sum / val_total_images

        writer.add_scalar('val/loss', val_loss, epoch + 1)
        writer.add_scalar('val/recon', val_recon, epoch + 1)
        writer.add_scalar('val/kl', val_kl, epoch + 1)
        # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch + 1)
        for dim, dim_kl in enumerate(avg_val_kl_per_dim):
            writer.add_scalar(f'val/kl_per_dim/dim_{dim}', dim_kl.item(), epoch + 1)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model._orig_mod.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, f"checkpoints/checkpoint_epoch{epoch+1}.pth")
            # model.eval()

            if val_preview_batch is not None:
                with torch.no_grad():
                    with autocast(device_type=device, dtype=torch.bfloat16):
                        mu, _ = model.encode(val_preview_batch)
                        val_recon_preview = model.decode(mu)

                original = val_preview_batch.float()
                reconstructed = torch.sigmoid(val_recon_preview).clamp(0, 1).float()
                comparison_grid = vutils.make_grid(torch.cat([original, reconstructed], dim=0), nrow=8)

                recon_path = f"checkpoints/val_reconstructions/epoch_{epoch+1:03d}.png"
                vutils.save_image(comparison_grid, recon_path)
                writer.add_image('val/reconstructions', comparison_grid, epoch + 1)
            
            if train_preview_batch is not None:
                with torch.no_grad():
                    with autocast(device_type=device, dtype=torch.bfloat16):
                        mu, _ = model.encode(train_preview_batch)
                        train_recon_preview = model.decode(mu)

                original = train_preview_batch.float()
                reconstructed = torch.sigmoid(train_recon_preview).clamp(0, 1).float()
                comparison_grid = vutils.make_grid(torch.cat([original, reconstructed], dim=0), nrow=8)

                recon_path = f"checkpoints/train_reconstructions/epoch_{epoch+1:03d}.png"
                vutils.save_image(comparison_grid, recon_path)
                writer.add_image('train/reconstructions', comparison_grid, epoch + 1)

            writer.flush()

    torch.save(model._orig_mod.state_dict(), "checkpoints/final_model.pth")
    writer.close()

if __name__ == '__main__':
    main()
