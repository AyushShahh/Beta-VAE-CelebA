# Beta-VAE with Controlled Capacity Increase (CelebA)

A **β-Variational Autoencoder (β-VAE)** trained on the **CelebA dataset** to learn disentangled latent representations of human faces. The model can generate faces, reconstruct images, and enable controllable latent editing of facial attributes such as smile, pose, and face shape.

<video src="./assets/video.mp4" autoplay loop controls width="800">
  Your browser does not support the video tag.
</video>

The model is trained on **64×64 CelebA images** with **capacity annealing** from the paper [Understanding Disentanglement in β-VAE](https://arxiv.org/pdf/1804.03599).

Changes were made to both the **architecture and training procedure** compared to the original implementation.

Model weights and direction tensors are available on [Hugging Face](https://huggingface.co/ayushshah/beta-vae-capacity-annealing-celeba).<br>
Run the demo on [Hugging Face Spaces](https://huggingface.co/spaces/ayushshah/Beta-VAE-Latent-Editing-CelebA).

# Features
* Face reconstruction
* Unconditional face generation
* Latent interpolation
* Feature extraction
* Controllable latent editing of facial attributes

The encoder maps images into a structured latent representation, while the decoder generates realistic face reconstructions and samples from latent vectors drawn from a Gaussian prior. The learned latent space exhibits partially disentangled factors corresponding to interpretable facial attributes such as:

* smiling
* face shape
* head rotation
* background color
* pose

# Architecture

The model follows the standard **encoder–decoder VAE structure**.

Encoder:

* Residual convolutional blocks
* Group Normalization
* Outputs mean and log-variance vectors

Decoder:

* Mirror architecture of the encoder
* Residual blocks with upsampling
* Generates reconstructed images from latent vectors

Leaky ReLU activations are used throughout the network to allow for better gradient flow and the weights are initialized using Kaiming initialization to facilitate training.

### Residual Blocks

Instead of simple convolutional blocks, **Residual Blocks** are used in both encoder and decoder. These skip connections improve gradient flow during training and allow deeper representations.

### Normalization

`BatchNorm` should be avoided in VAEs because it interferes with the latent distribution statistics.

Instead, **Group Normalization** is used to stabilize training.

Without normalization, training becomes unstable and may result in:
* exploding gradients
* NaNs in loss

# Training
The model is trained using the **capacity annealing objective** proposed in the β-VAE paper.

Instead of enforcing strong KL regularization from the start, the KL divergence capacity is gradually increased during training. This allows the model to first focus on reconstruction quality before encouraging disentanglement.

Loss function:

```
L = reconstruction_loss + γ | KL − C |
```

Where:

* **γ** controls how strongly KL divergence matches the capacity
* **C** is the target capacity that increases during training

# Hyperparameters
- Latent Dimensions: 32
- C_max: 50
- Gamma: 1000
- Epochs: 200
- Capacity Increase Schedule: 80 epochs
- Batch Size: 128
- Optimizer: AdamW
- Weight Decay: 0
- Learning Rate: ≤ 3e-5
- Loss: BCE with logits (mean over batch)

Higher learning rates lead to unstable training and exploding gradients.

# Dataset
Dataset used: [jessicali9530/celeba-dataset (Kaggle)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Preprocessing pipeline:

```
CenterCrop(148)
Resize(64x64)
ToTensor()
```

# Latent Directions

The Hugging Face repository contains **latent direction tensors** for semantic facial attributes.

These are computed using the difference between latent means of random positive and negative attribute samples from CelebA annotations (`10,000` positive and negative samples each).

Example:

```
direction = mean(latents_smiling) − mean(latents_not_smiling)
```

Each direction tensor has shape:

```
(32,)
```

which matches the dimensionality of the latent space.

Latent editing is performed by shifting the latent vector:

```
edited_latent = latent + strength * direction
```

where `strength` controls how strongly the attribute is applied.

Positive values increase the attribute while negative values decrease it.


# Usage

To run the model:

1. Clone this repository

```bash
git clone https://github.com/AyushShahh/Beta-VAE-CelebA
cd Beta-VAE-CelebA
```

2. Create a virtual environment using Conda or any other environment manager of your choice and install the required dependencies:

```bash
conda create -n beta-vae-celeba python=3.12
conda activate beta-vae-celeba
pip install -r requirements.txt
```

3. Run this script to download the model weights and direction tensors from Hugging Face and place them in the appropriate directories (for inference):

```bash
python download.py
```

4. If you have not trained the model yourself and are using the provided weights, you must change the `checkpoint_path` variable in `app.py` and `test.py` to point to the downloaded weights (`./checkpoints/model.safetensors`).

5. Launch the Gradio demo:

```bash
python app.py
```

The repository contains the **architecture implementation and inference pipeline**, so the model can be loaded using the provided architecture file.

Typical workflow:

1. Encode an image to obtain its latent vector
2. Add a scaled direction tensor
3. Decode the modified latent vector

This enables interactive editing of facial attributes.

You can also:
- sample random latent vectors from a standard normal distribution and decode them to generate new faces.
- sample nearby latent vectors to generate similar faces with small variations.
- perform linear interpolation between two latent vectors to create smooth transitions between faces.

# Observations

- Active Dimensions (kl > 0.1): 19
- Active Dimensions (kl > 1.0): 17

These correspond to latent variables that capture meaningful variations in the dataset.

# License

Please refer to the dataset license for CelebA usage restrictions.<br>
The code in this repository is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
