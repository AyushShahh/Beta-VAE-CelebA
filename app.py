import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from architecture import CelebAVAE
from safetensors.torch import load_file

# Setup device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CelebAVAE(latent_dim=32).to(device)

# Load checkpoint handling compiled models
checkpoint_path = 'checkpoints/final_model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
if checkpoint_path.endswith('.safetensors'):
    checkpoint = load_file(checkpoint_path, device=device)
model.load_state_dict(checkpoint)
model.eval()

# Load available directions
directions_dir = "directions"
available_directions = ["None"]
if os.path.exists(directions_dir):
    available_directions += sorted([f.replace('.pt', '') for f in os.listdir(directions_dir) if f.endswith('.pt')])

# Dataset path for index selection
dataset_path = "./dataset/img_align_celeba/img_align_celeba"
image_files = sorted(os.listdir(dataset_path)) if os.path.exists(dataset_path) else []

def encode_image(input_image, dataset_index):
    # Determine the input source
    if input_image is not None:
        img = input_image.convert("RGB")
        # Crop to square if needed
        w, h = img.size
        if w != h:
            min_dim = min(w, h)
            img = transforms.CenterCrop(min_dim)(img)
        img = transforms.Resize((64, 64))(img)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    elif dataset_index is not None and image_files:
        idx = int(dataset_index)
        if idx < 0 or idx >= len(image_files):
            return None, None, None
        img_name = os.path.join(dataset_path, image_files[idx])
        img = Image.open(img_name).convert("RGB")
        # Apply dataset transform match from train.py
        img = transforms.CenterCrop(148)(img)
        img = transforms.Resize((64, 64))(img)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    else:
        return None, None, None

    with torch.no_grad():
        mu, logvar = model.encode(x)
        # Base reconstruction from mu
        recon = model.decode(mu)

    # Convert tensors back to PIL Images and scale up to 128x128
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)
    
    orig_img = resize(to_pil(x.squeeze(0).cpu()))
    recon_img = resize(to_pil(torch.sigmoid(recon).squeeze(0).cpu()))
    
    return orig_img, recon_img, mu

def edit_latent(mu, feature_name, strength):
    if mu is None:
        return None
        
    with torch.no_grad():
        # Edited reconstruction
        if feature_name != "None":
            direction_path = os.path.join(directions_dir, f"{feature_name}.pt")
            if os.path.exists(direction_path):
                direction = torch.load(direction_path, map_location=device, weights_only=True)
                direction = direction.to(device)
                edit_mu = mu + strength * direction
                edited = model.decode(edit_mu)
            else:
                edited = model.decode(mu)
        else:
            edited = model.decode(mu)

    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)
    edited_img = resize(to_pil(torch.sigmoid(edited).squeeze(0).cpu()))
    
    return edited_img

# Gradio Interface
with gr.Blocks(title="Beta-VAE Latent Editing") as demo:
    gr.Markdown("# Beta-VAE Latent Space Editing")
    gr.Markdown("Upload an image (will be cropped to square) or select a dataset index to see its reconstruction and edit its features.")
    
    # State to hold the latent vector
    latent_state = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image (Upload)")
            dataset_index = gr.Number(label="Or Dataset Index", value=0, precision=0, minimum=0, maximum=len(image_files)-1 if image_files else 0)
            
            with gr.Row():
                load_btn = gr.Button("Load & Reconstruct")
                clear_btn = gr.Button("Clear")
            
            gr.Markdown("### Edit Dimensions")
            feature_dropdown = gr.Dropdown(choices=available_directions, value="None", label="Feature to Edit")
            strength_slider = gr.Slider(minimum=-3.0, maximum=3.0, value=0.0, step=0.1, label="Edit Strength")
            gr.Markdown("Positive values increase the attribute, while negative values reduce it.")
            
        with gr.Column():
            gr.Markdown("### Original (Left) & Reconstruction (Right)")
            with gr.Row():
                orig_output = gr.Image(type="pil")
                recon_output = gr.Image(type="pil")
            
            gr.Markdown("### Edited Image")
            edited_output = gr.Image(type="pil")

    # 1. Load the image and reconstruct it. Saves the latent state.
    load_btn.click(
        fn=encode_image,
        inputs=[input_image, dataset_index],
        outputs=[orig_output, recon_output, latent_state]
    ).then(
        fn=edit_latent,
        inputs=[latent_state, feature_dropdown, strength_slider],
        outputs=[edited_output]
    )

    # 2. Edit the image using the saved latent state.
    feature_dropdown.change(
        fn=edit_latent,
        inputs=[latent_state, feature_dropdown, strength_slider],
        outputs=[edited_output]
    )
    
    strength_slider.change(
        fn=edit_latent,
        inputs=[latent_state, feature_dropdown, strength_slider],
        outputs=[edited_output]
    )

    # 3. Clear button resets all states and outputs.
    clear_btn.click(
        fn=lambda: (None, 0, "None", 0.0, None, None, None, None),
        inputs=[],
        outputs=[
            input_image, dataset_index, feature_dropdown, strength_slider,
            orig_output, recon_output, edited_output, latent_state
        ]
    )

if __name__ == "__main__":
    demo.launch()
