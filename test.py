from architecture import CelebAVAE
import torch
import matplotlib.pyplot as plt
from dataset import CelebADataset
from torchvision import transforms
from torchvision.utils import make_grid


transform = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = 'cuda'

model = CelebAVAE(latent_dim=32).to('cuda')
state_dict = torch.load('checkpoints/checkpoint_epoch200.pth', map_location=device)['model_state_dict']

model.load_state_dict(state_dict)
model.eval()

def slerp(z1, z2, t):
    """
    Spherical linear interpolation between z1 and z2
    z1, z2: (1, latent_dim)
    t: scalar between 0 and 1
    """
    z1_norm = z1 / torch.norm(z1)
    z2_norm = z2 / torch.norm(z2)

    dot = torch.clamp(torch.sum(z1_norm * z2_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    if so == 0:
        return (1 - t) * z1 + t * z2

    return (
        torch.sin((1 - t) * omega) / so * z1
        + torch.sin(t * omega) / so * z2
    )

def random_sample():
    with torch.no_grad():
        z = torch.randn(1, 32).to('cuda')
        logits = model.decode(z)
        logits = torch.sigmoid(logits)

    logits = logits.cpu().squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(logits)
    plt.show()

def latent_interpolation(i1, i2):
    x1 = CelebADataset('dataset/img_align_celeba/img_align_celeba', i1, i1, transform=transform)[0]
    x2 = CelebADataset('dataset/img_align_celeba/img_align_celeba', i2, i2, transform=transform)[0]

    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)

    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

    steps = 15
    alphas = torch.linspace(0, 1, steps)

    z_interp = torch.stack([
        slerp(mu1, mu2, a).squeeze(0)
        for a in alphas
    ]).to(device)

    with torch.no_grad():
        logits = model.decode(z_interp)
        logits = torch.sigmoid(logits)

    logits = logits.cpu().permute(0, 2, 3, 1).numpy()

    grid = make_grid(torch.from_numpy(logits).permute(0, 3, 1, 2), nrow=steps)

    plt.figure(figsize=(steps * 2, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def latent_traversal(idx, dim):
    x1 = CelebADataset('dataset/img_align_celeba/img_align_celeba', idx, idx, transform=transform)[0]
    x1 = x1.unsqueeze(0).to(device)

    with torch.no_grad():
        mu1, _ = model.encode(x1)

    steps = 15

    # traversal relative to encoded latent value
    edit_offsets = torch.linspace(-3, 3, steps, device=device)

    z_edit = mu1.repeat(steps, 1)
    z_edit[:, dim] = mu1[0, dim] + edit_offsets
    # z_edit[:, dim] = edit_offsets

    with torch.no_grad():
        logits = model.decode(z_edit)
        logits = torch.sigmoid(logits)

    logits = logits.cpu().permute(0, 2, 3, 1).numpy()

    grid = make_grid(torch.from_numpy(logits).permute(0, 3, 1, 2), nrow=steps)

    plt.figure(figsize=(steps * 2, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def latent_manifold(latent_dim=32, steps=9, value_range=3.0):
    model.eval()

    strong = [0, 2, 3, 4, 5, 6, 7, 8, 13, 14, 20, 21, 24, 26, 27, 28, 29]
    values = torch.linspace(-value_range, value_range, steps, device=device)

    images = []

    with torch.no_grad():
        for dim in strong:
            z = torch.zeros(steps, latent_dim, device=device)

            z[:, dim] = values

            logits = model.decode(z)
            imgs = torch.sigmoid(logits)

            images.append(imgs.cpu())

    images = torch.cat(images, dim=0)

    grid = make_grid(images, nrow=steps)

    plt.figure(figsize=(steps * 1.5, latent_dim * 1.5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # 0 - smile and glasses
    # 2 - not sure
    # 3 - Rotation + Gender sometimes
    # 4 - Background colour
    # 5 - Skin tone (Background color, hair)
    # 6 - hairs
    # 7 - depends on image
    # 8 - depends on image (bangs / skin tone + glasses)
    # 13 - not specific
    # 14 - not sure
    # middle not sure
    # 24 - smile
    # 26 - background color
    # 27 - maybe hat or hair
    # 28 - face shape
    # 29 - not sure (smile, hairs maybe)
    latent_traversal(7, 3)
    # latent_manifold()