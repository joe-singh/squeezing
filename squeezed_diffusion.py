import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import linalg
import random
import copy
from sklearn.neighbors import NearestNeighbors
import wandb 

# Fail early if WANDB_API_KEY not set
if not os.environ.get("WANDB_API_KEY"):
    print(
        "Error: WANDB_API_KEY not found in environment.\n"
        "Set it with:\n"
        "    export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "or run:\n"
        "    wandb login"
    )
    print("Continuing without wandb logging")

# Import from diffusers
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from diffusers.utils import make_image_grid
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKL

# Import for FID
from torchvision.models import inception_v3
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# Import for PR from torch-fidelity
from torch_fidelity import calculate_metrics

# Your dataset class
from torchvision.datasets import CIFAR10

class CIFAR10ImagesOnly(CIFAR10):
    """Standard CIFAR-10 but __getitem__ returns only the image."""
    def __init__(self, root, train=True, transform=None, download=True, keep_classes=None):
        super().__init__(root=root, train=train, transform=transform, download=download)

        if keep_classes is not None:
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
            keep_idx = [class_to_idx[c] for c in keep_classes]
            mask = np.isin(self.targets, keep_idx)
            self.data = self.data[mask]
            self.targets = [t for t in np.array(self.targets)[mask]]
            print(f"Keeping classes {keep_classes} → {len(self.data)} images total")

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img


# FID Score Calculation Classes and Functions

class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID) score using standard pytorch-fid approach"""

    def __init__(self, device='cuda', dims=2048):
        self.device = device
        self.dims = dims

        # Initialize InceptionV3 with standard FID settings
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()

        # Store history
        self.fid_history = []
        self.epoch_history = []

        # Pre-computed statistics for common datasets
        # You can download these from pytorch-fid repository
        self.precomputed_stats = {}

    @torch.no_grad()
    def extract_features(self, images, batch_size=50, show_bar=False):
        """Extract features from images using Inception v3"""
        self.inception.eval()

        features_list = []

        # Process in batches
        n_batches = (len(images) + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches),
                      desc="Extracting features",
                      disable=not show_bar,
                      leave=False):

            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))
            batch = images[start:end].to(self.device)

            # Images should be in [0, 1] range
            if batch.min() < 0:
                batch = (batch + 1) / 2  # Convert from [-1, 1] to [0, 1]

            features = self.inception(batch)
            # InceptionV3 from pytorch_fid returns a list, extract the first (and only) element
            features = features[0].squeeze(3).squeeze(2)  # Remove spatial dimensions
            features_list.append(features.cpu())

        return torch.cat(features_list, dim=0).numpy()

    def calculate_statistics(self, features):
        """Calculate mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real_images, generated_images, batch_size=50):
        """
        Calculate FID score between real and generated images

        Args:
            real_images: Tensor of real images
            generated_images: Tensor of generated images
            batch_size: Batch size for feature extraction

        Returns:
            FID score (lower is better)
        """
        # Extract features
        print("Extracting features from real images for FID...")
        real_features = self.extract_features(real_images, batch_size)

        print("Extracting features from generated images for FID...")
        gen_features = self.extract_features(generated_images, batch_size)

        # Calculate statistics
        print("Calculating statistics for FID...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)

        # Calculate FID
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        return fid_score

    def update_history(self, epoch, fid_score):
        """Update FID history"""
        self.epoch_history.append(epoch)
        self.fid_history.append(fid_score)

    def plot_history(self, save_path=None, title="FID Score History"):
        """Plot FID score history"""
        if not self.fid_history:
            print("No FID history to plot")
            return

        plt.figure(figsize=(10, 6))

        # Plot line with markers
        plt.plot(self.epoch_history, self.fid_history, 'b-o', linewidth=2, markersize=6)

        # Add value labels on each point
        for i, (epoch, fid) in enumerate(zip(self.epoch_history, self.fid_history)):
            plt.annotate(f'{fid:.2f}',
                        (epoch, fid),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)

        # Find and mark best score
        best_fid = min(self.fid_history)
        best_epoch = self.epoch_history[self.fid_history.index(best_fid)]

        # Add horizontal line for best score
        plt.axhline(y=best_fid, color='r', linestyle='--', alpha=0.7)
        plt.text(self.epoch_history[0], best_fid - 5, f'Best: {best_fid:.2f} (Epoch {best_epoch})',
                color='r', fontsize=12, va='top')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('FID Score (lower is better)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)

        # Set y-axis to start slightly below minimum FID for better visualization
        y_min = min(self.fid_history) * 0.9
        y_max = max(self.fid_history) * 1.1
        plt.ylim(y_min, y_max)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"FID plot saved to {save_path}")

        plt.close()

    def save_history(self, save_path):
        """Save FID history to a text file"""
        with open(save_path, 'w') as f:
            f.write("Epoch,FID Score\n")
            for epoch, fid in zip(self.epoch_history, self.fid_history):
                f.write(f"{epoch},{fid:.4f}\n")


# Inception Score Calculator
class InceptionScoreCalculator:
    """Calculate Inception Score (IS) using InceptionV3"""

    def __init__(self, device='cuda', resize=True, splits=10):
        self.device = device
        self.resize = resize
        self.splits = splits

        # Load pre-trained InceptionV3
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

        # Store history
        self.is_history = []
        self.is_std_history = []
        self.epoch_history = []

        # For CIFAR-10, we need to resize to 299x299 for InceptionV3
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    @torch.no_grad()
    def get_predictions(self, images, batch_size=32):
        """Get softmax predictions from InceptionV3"""
        self.inception.eval()

        predictions = []
        n_batches = (len(images) + batch_size - 1) // batch_size

        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        for i in tqdm(range(n_batches), desc="Getting predictions", leave=False):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))
            batch = images[start:end].to(self.device)

            # Convert to [0, 1] range if needed
            if batch.min() < 0:
                batch = (batch + 1) / 2

            # Resize if needed (CIFAR-10 is 32x32, InceptionV3 expects 299x299)
            if self.resize and batch.shape[2] != 299:
                batch = self.up(batch)

            # Apply ImageNet normalization
            batch = (batch - mean) / std

            # Get predictions
            pred = self.inception(batch)
            pred = F.softmax(pred, dim=1).cpu().numpy()
            predictions.append(pred)

        return np.concatenate(predictions, axis=0)

    def calculate_inception_score(self, predictions, splits=10):
        """
        Calculate Inception Score from predictions

        Args:
            predictions: numpy array of softmax predictions
            splits: number of splits for calculating IS

        Returns:
            mean_is: mean inception score
            std_is: standard deviation of inception score
        """
        # Split predictions
        n = predictions.shape[0]
        n = (n // splits) * splits     # discard the last (n % splits) samples
        predictions = predictions[:n]  # keep only full splits
        split_size = n // splits
        scores = []

        for i in range(splits):
            start = i * split_size
            end = (i + 1) * split_size if i < splits - 1 else n
            preds_split = predictions[start:end]

            # Calculate p(y|x)
            p_yx = preds_split

            # Calculate p(y)
            p_y = np.mean(p_yx, axis=0, keepdims=True)

            # Calculate KL divergence
            kl_d = p_yx * (np.log(p_yx + 1e-8) - np.log(p_y + 1e-8))
            kl_d = np.sum(kl_d, axis=1)

            # Calculate IS for this split
            is_score = np.exp(np.mean(kl_d))
            scores.append(is_score)

        return np.mean(scores), np.std(scores)

    def calculate_is(self, generated_images, batch_size=32):
        """
        Calculate Inception Score for generated images

        Args:
            generated_images: Tensor of generated images
            batch_size: Batch size for processing

        Returns:
            mean_is: mean inception score
            std_is: standard deviation
        """
        print("Calculating Inception Score...")

        # Get predictions
        predictions = self.get_predictions(generated_images, batch_size)

        # Calculate IS
        mean_is, std_is = self.calculate_inception_score(predictions, self.splits)

        return mean_is, std_is

    def update_history(self, epoch, mean_is, std_is):
        """Update IS history"""
        self.epoch_history.append(epoch)
        self.is_history.append(mean_is)
        self.is_std_history.append(std_is)

    def plot_history(self, save_path=None, title="Inception Score History"):
        """Plot IS history with error bars"""
        if not self.is_history:
            print("No IS history to plot")
            return

        plt.figure(figsize=(10, 6))

        # Plot with error bars
        plt.errorbar(self.epoch_history, self.is_history, yerr=self.is_std_history,
                    fmt='b-o', linewidth=2, markersize=6, capsize=5)

        # Add value labels
        for i, (epoch, is_score) in enumerate(zip(self.epoch_history, self.is_history)):
            plt.annotate(f'{is_score:.2f}',
                        (epoch, is_score),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)

        # Find and mark best score
        best_is = max(self.is_history)
        best_epoch = self.epoch_history[self.is_history.index(best_is)]

        # Add horizontal line for best score
        plt.axhline(y=best_is, color='r', linestyle='--', alpha=0.7)
        plt.text(self.epoch_history[0], best_is + 0.1, f'Best: {best_is:.2f} (Epoch {best_epoch})',
                color='r', fontsize=12, va='bottom')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Inception Score (higher is better)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"IS plot saved to {save_path}")

        plt.close()

    def save_history(self, save_path):
        """Save IS history to a text file"""
        with open(save_path, 'w') as f:
            f.write("Epoch,IS Mean,IS Std\n")
            for epoch, mean_is, std_is in zip(self.epoch_history, self.is_history, self.is_std_history):
                f.write(f"{epoch},{mean_is:.4f},{std_is:.4f}\n")


# NEW: Precision and Recall Score Calculator using torch-fidelity
class PrecisionRecallCalculator:
    """Calculate Precision and Recall (PR) scores using torch-fidelity."""

    def __init__(self, device='cuda', k=3):
        self.device = device
        self.k = k # Number of neighbors for k-NN

        # Store history
        self.precision_history = []
        self.recall_history = []
        self.epoch_history = []

    def calculate_pr(self, real_images_path, generated_images_path):
        """
        Compute Precision & Recall with torch-fidelity.
        """
        import torch_fidelity
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=real_images_path,
            input2=generated_images_path,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
            prc=True,
            prc_nearest_k=self.k,
            verbose=False, # <--- THE CRUCIAL CHANGE
        )

        print("DEBUG: Dictionary returned by torch_fidelity:", metrics_dict)

        precision = metrics_dict['precision']
        recall = metrics_dict['recall']

        return precision, recall

    def update_history(self, epoch, precision, recall):
        """Update PR history"""
        self.epoch_history.append(epoch)
        self.precision_history.append(precision)
        self.recall_history.append(recall)

    def plot_history(self, save_path=None, title="Precision and Recall History"):
        """Plot PR history"""
        if not self.precision_history:
            print("No PR history to plot")
            return

        plt.figure(figsize=(10, 6))

        plt.plot(self.epoch_history, self.precision_history, 'b-o', linewidth=2, markersize=6, label='Precision')
        plt.plot(self.epoch_history, self.recall_history, 'g-o', linewidth=2, markersize=6, label='Recall')

        # Add value labels
        for i, (epoch, p, r) in enumerate(zip(self.epoch_history, self.precision_history, self.recall_history)):
            plt.annotate(f'P:{p:.2f}', (epoch, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(f'R:{r:.2f}', (epoch, r), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)


        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()

        # Set y-axis limits
        plt.ylim(0, 1.05)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR plot saved to {save_path}")

        plt.close()


    def save_history(self, save_path):
        """Save PR history to a text file"""
        with open(save_path, 'w') as f:
            f.write("Epoch,Precision,Recall\n")
            for epoch, p, r in zip(self.epoch_history, self.precision_history, self.recall_history):
                f.write(f"{epoch},{p:.4f},{r:.4f}\n")


from pathlib import Path
REAL_STATS = Path("cifar10_inception_v3_stats.npz")

@torch.no_grad()
def get_real_stats(fid_calc, dataloader, device, batch_size):
    if REAL_STATS.exists():
        data = np.load(REAL_STATS)
        return data["mu"], data["sigma"]

    print("Computing CIFAR-10 stats with FID Inception …")
    feats = []
    for batch in tqdm(dataloader, desc="features"):
        feats.append(fid_calc.extract_features(batch.to(device),
                                               batch_size=batch_size,
                                               show_bar=False)
        )
    feats = np.concatenate(feats, 0)        # shape (50 000, 2048)

    mu, sigma = fid_calc.calculate_statistics(feats)
    np.savez(REAL_STATS, mu=mu, sigma=sigma)
    return mu, sigma


def evaluate_fid_score(model,
                       scheduler,
                       dataloader,          # CIFAR-10 dataloader (train split)
                       fid_calculator,
                       config,
                       epoch,
                       device):
    """
    Generate samples and compute FID score using standard Inception v3 features.
    This ensures compatibility with FID scores reported in other papers.
    """

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gen = torch.Generator(device).manual_seed(seed)

    model.eval()

    # Config shortcuts
    n_gen = config.get("fid_n_generated", 10000)
    gen_batch = config["eval_batch_size"]
    feat_batch = config["fid_batch_size"]
    n_steps_eval = config.get("fid_inference_steps", 50)

    print(f"\n[Epoch {epoch}] FID: generating {n_gen} samples "
          f"({n_steps_eval} DDPM steps each)...")

    # 1. Generate images
    generated = []
    scheduler.set_timesteps(n_steps_eval)

    with torch.no_grad():
        for _ in tqdm(range((n_gen + gen_batch - 1) // gen_batch),
                      desc="Generating"):
            remaining = n_gen - sum(t.size(0) for t in generated)
            bsz = min(gen_batch, remaining)
            shape = (bsz, 3, config["image_size"], config["image_size"])

            # Start from noise (with optional squeezing)
            noise = torch.randn(shape, device=device, generator=gen)
            if (scheduler.noise_squeezer is not None
                and scheduler.noise_squeezer.squeeze_strength != 0):
                t_max = scheduler.config.num_train_timesteps - 1
                S = scheduler.noise_squeezer.get_squeeze_matrix(
                        noise,
                        t=torch.full((bsz,), t_max, device=device),
                        scheduler=scheduler)
                img = scheduler.noise_squeezer.apply_squeeze(noise, S)
            else:
                img = noise

            # Denoising loop
            for t in scheduler.timesteps:
                t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
                eps = model(img, t_batch).sample
                img = scheduler.step(eps, t, img, generator=gen).prev_sample

            generated.append(img.cpu())

    generated = torch.cat(generated, 0)[:n_gen]  # [N,3,H,W]

    # 2. Extract features from generated images
    print("Extracting Inception features from generated samples for FID...")
    gen_feats = fid_calculator.extract_features(generated, batch_size=feat_batch)
    mu_gen, sigma_gen = fid_calculator.calculate_statistics(gen_feats)

    # 3. Get cached CIFAR-10 statistics (computed with same Inception model)
    mu_real, sigma_real = get_real_stats(fid_calculator,
                                         dataloader,
                                         device,
                                         batch_size=feat_batch)

    # 4. Calculate FID
    fid_score = calculate_frechet_distance(
                    mu_real, sigma_real, mu_gen, sigma_gen)

    # 5. Update history and save plots
    fid_calculator.update_history(epoch, fid_score)

    plot_path = os.path.join(config["output_dir"], "fid_history.png")
    fid_calculator.plot_history(plot_path)

    history_path = os.path.join(config["output_dir"], "fid_history.txt")
    fid_calculator.save_history(history_path)

    print(f"[Epoch {epoch}] FID = {fid_score:.4f}")

    model.train()
    return fid_score


# Inception Score evaluation function
def evaluate_inception_score(model,
                           scheduler,
                           is_calculator,
                           config,
                           epoch,
                           device):
    """
    Generate samples and compute Inception Score.
    """

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gen = torch.Generator(device).manual_seed(seed)

    model.eval()

    # Config shortcuts
    n_gen = config.get("is_n_generated", 10000)
    gen_batch = config["eval_batch_size"]
    n_steps_eval = config.get("is_inference_steps", 50)

    print(f"\n[Epoch {epoch}] IS: generating {n_gen} samples "
          f"({n_steps_eval} DDPM steps each)...")

    # 1. Generate images
    generated = []
    scheduler.set_timesteps(n_steps_eval)

    with torch.no_grad():
        for _ in tqdm(range((n_gen + gen_batch - 1) // gen_batch),
                      desc="Generating"):
            remaining = n_gen - sum(t.size(0) for t in generated)
            bsz = min(gen_batch, remaining)
            shape = (bsz, 3, config["image_size"], config["image_size"])

            # Start from noise (with optional squeezing)
            noise = torch.randn(shape, device=device, generator=gen)
            if (scheduler.noise_squeezer is not None
                and scheduler.noise_squeezer.squeeze_strength != 0):
                t_max = scheduler.config.num_train_timesteps - 1
                S = scheduler.noise_squeezer.get_squeeze_matrix(
                        noise,
                        t=torch.full((bsz,), t_max, device=device),
                        scheduler=scheduler)
                img = scheduler.noise_squeezer.apply_squeeze(noise, S)
            else:
                img = noise

            # Denoising loop
            for t in scheduler.timesteps:
                t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
                eps = model(img, t_batch).sample
                img = scheduler.step(eps, t, img, generator=gen).prev_sample

            generated.append(img.cpu())

    generated = torch.cat(generated, 0)[:n_gen]  # [N,3,H,W]

    # 2. Calculate Inception Score
    mean_is, std_is = is_calculator.calculate_is(generated, batch_size=config["is_batch_size"])

    # 3. Update history and save plots
    is_calculator.update_history(epoch, mean_is, std_is)

    plot_path = os.path.join(config["output_dir"], "is_history.png")
    is_calculator.plot_history(plot_path)

    history_path = os.path.join(config["output_dir"], "is_history.txt")
    is_calculator.save_history(history_path)

    print(f"[Epoch {epoch}] IS = {mean_is:.4f} ± {std_is:.4f}")

    model.train()
    return mean_is, std_is


# NEW: Precision and Recall evaluation function using torch-fidelity
def evaluate_pr_score(model,
                      scheduler,
                      dataloader_eval, # Dataloader for real images (eval split recommended)
                      pr_calculator,
                      config,
                      epoch,
                      device):
    """
    Generate samples and compute Precision and Recall scores using torch-fidelity.
    """

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gen = torch.Generator(device).manual_seed(seed)

    model.eval()

    # Config shortcuts
    n_real = config.get("pr_n_real", 10000)
    n_gen = config.get("pr_n_generated", 10000)
    gen_batch = config["eval_batch_size"]
    n_steps_eval = config.get("pr_inference_steps", 50)

    print(f"\n[Epoch {epoch}] PR: generating {n_gen} samples "
          f"({n_steps_eval} DDPM steps each) and using {n_real} real samples...")

    # 1. Generate images and save them to a temporary directory
    generated_dir = os.path.join(config["output_dir"], "generated_pr")
    os.makedirs(generated_dir, exist_ok=True)

    generated_count = 0
    scheduler.set_timesteps(n_steps_eval)

    with torch.no_grad():
        for _ in tqdm(range((n_gen + gen_batch - 1) // gen_batch),
                      desc="Generating"):
            remaining = n_gen - generated_count
            bsz = min(gen_batch, remaining)
            shape = (bsz, 3, config["image_size"], config["image_size"])

            # Start from noise (with optional squeezing)
            noise = torch.randn(shape, device=device, generator=gen)
            if (scheduler.noise_squeezer is not None
                and scheduler.noise_squeezer.squeeze_strength != 0):
                t_max = scheduler.config.num_train_timesteps - 1
                S = scheduler.noise_squeezer.get_squeeze_matrix(
                        noise,
                        t=torch.full((bsz,), t_max, device=device),
                        scheduler=scheduler)
                img = scheduler.noise_squeezer.apply_squeeze(noise, S)
            else:
                img = noise


            # Denoising loop
            for t in scheduler.timesteps:
                t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
                eps = model(img, t_batch).sample
                img = scheduler.step(eps, t, img, generator=gen).prev_sample

            # Denormalize and save generated images
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img.cpu().permute(0, 2, 3, 1).numpy()
            for i in range(img.shape[0]):
                pil_img = Image.fromarray((img[i] * 255).astype(np.uint8))
                pil_img.save(os.path.join(generated_dir, f"gen_{generated_count:05d}.png"))
                generated_count += 1




    # 2. Save real images to a temporary directory
    real_dir = os.path.join(config["output_dir"], "real_pr")
    os.makedirs(real_dir, exist_ok=True)

    real_count = 0
    with torch.no_grad():
      for i, batch in enumerate(dataloader_eval):
        if real_count >= n_real:
          break
        batch = (batch / 2 + 0.5).clamp(0, 1) # Denormalize
        batch = batch.cpu().permute(0, 2, 3, 1).numpy()
        for j in range(batch.shape[0]):
            if real_count >= n_real:
                break
            pil_img = Image.fromarray((batch[j] * 255).astype(np.uint8))
            pil_img.save(os.path.join(real_dir, f"real_{real_count:05d}.png"))
            real_count += 1

    # 3. Calculate Precision and Recall using torch-fidelity
    precision, recall = pr_calculator.calculate_pr(
        real_images_path=real_dir,
        generated_images_path=generated_dir
    )

    # 4. Update history and save plots
    pr_calculator.update_history(epoch, precision, recall)

    plot_path = os.path.join(config["output_dir"], "pr_history.png")
    pr_calculator.plot_history(plot_path)

    history_path = os.path.join(config["output_dir"], "pr_history.txt")
    pr_calculator.save_history(history_path)

    print(f"[Epoch {epoch}] PR = P: {precision:.4f}, R: {recall:.4f}")

    # Clean up temporary directories
    shutil.rmtree(generated_dir) # Uncomment to remove temporary files
    shutil.rmtree(real_dir)     # Uncomment to remove temporary files

    model.train()
    return precision, recall


class NoiseSqueezer:
    """Handles PCA-based noise squeezing for DDPM."""

    def __init__(self, squeeze_strength=0.0, quantum_limited=False):
        self.squeeze_strength = squeeze_strength
        self.quantum_limited = quantum_limited
        self.eigenvalues = None
        self.eigenvectors = None
        self.covariance_matrix = None

    def calculate_dataset_statistics(self, dataloader, device, max_samples=10000):
        """Calculate PCA components from dataset."""
        print("Calculating dataset covariance for PCA-based squeezing...")

        # Running sums
        sum_rgb = torch.zeros(3, device=device)
        sum_outer = torch.zeros(3, 3, device=device)
        n_pixels = 0
        img_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing PCA"):
                if max_samples and img_count >= max_samples:
                    break

                batch = batch.to(device)
                B, C, H, W = batch.shape

                if max_samples and img_count + B > max_samples:
                    batch = batch[:max_samples - img_count]
                    B = batch.shape[0]

                img_count += B
                pixels = batch.permute(1, 0, 2, 3).reshape(3, -1)

                sum_rgb += pixels.sum(dim=1)
                sum_outer += pixels @ pixels.t()
                n_pixels += pixels.shape[1]

        # Mean and covariance
        mean = sum_rgb / n_pixels
        cov = sum_outer / n_pixels - mean[:, None] * mean[None, :]

        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(cov)

        self.covariance_matrix = cov
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs

        print(f"Used {img_count} images → {n_pixels:,} pixels")
        print("Eigenvalues (ascending):", eigvals.tolist())

        return cov, eigvals, eigvecs

    def get_squeeze_matrix(self, x, t=None, scheduler=None):
        """Get squeeze transformation matrix."""
        B = x.shape[0]
        device = x.device
        identity = torch.eye(3, device=device)

        # Principal direction (largest eigenvalue)
        principal_direction = self.eigenvectors[:, -1]

        # Projection matrices
        projection = torch.outer(principal_direction, principal_direction)
        orthogonal_projection = identity - projection

        # Time-dependent squeezing if scheduler provided
        if t is not None and scheduler is not None:
            # Get betas from scheduler
            betas = scheduler.betas.to(device)
            beta_max = betas.max()

            # Get beta values for the current timesteps
            if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
                # Single timestep
                beta_t = betas[t]
            else:
                # Batch of timesteps
                beta_t = betas[t]

            # Scale squeeze strength by beta_t / beta_max
            # At early timesteps (low t): beta_t is small → less squeezing
            # At late timesteps (high t): beta_t approaches beta_max → more squeezing
            time_scale = (beta_t / beta_max).view(-1, 1, 1)
            squeeze_param = self.squeeze_strength * time_scale
        else:
            squeeze_param = self.squeeze_strength

        if self.quantum_limited:
            # Volume-preserving squeeze
            n = 3  # RGB dimensions
            r = 1.0
            q = r / (n - 1)

            if isinstance(squeeze_param, torch.Tensor) and squeeze_param.dim() > 0:
                squeeze_matrices = []
                for i in range(B):
                    param = squeeze_param[i].item() if i < squeeze_param.shape[0] else squeeze_param[-1].item()
                    squeeze_factor = torch.exp(torch.tensor(-r * param, device=device))
                    antisqueeze_factor = torch.exp(torch.tensor(q * param, device=device))
                    matrix = squeeze_factor * projection + antisqueeze_factor * orthogonal_projection
                    squeeze_matrices.append(matrix)
                return torch.stack(squeeze_matrices)
            else:
                squeeze_factor = torch.exp(torch.tensor(-r * squeeze_param, device=device))
                antisqueeze_factor = torch.exp(torch.tensor(q * squeeze_param, device=device))
                matrix = squeeze_factor * projection + antisqueeze_factor * orthogonal_projection
                return matrix.unsqueeze(0).repeat(B, 1, 1)
        else:
            # Simple squeeze (not volume-preserving)
            if isinstance(squeeze_param, torch.Tensor) and squeeze_param.dim() > 0:
                squeeze_matrices = []
                for i in range(B):
                    param = squeeze_param[i].item() if i < squeeze_param.shape[0] else squeeze_param[-1].item()
                    squeeze_factor = torch.exp(torch.tensor(-param, device=device))
                    matrix = squeeze_factor * projection + orthogonal_projection
                    squeeze_matrices.append(matrix)
                return torch.stack(squeeze_matrices)
            else:
                squeeze_factor = torch.exp(torch.tensor(-squeeze_param, device=device))
                matrix = squeeze_factor * projection + orthogonal_projection
                return matrix.unsqueeze(0).repeat(B, 1, 1)

    def apply_squeeze(self, noise, squeeze_matrices):
        """Apply squeeze transformation to noise."""
        B, C, H, W = noise.shape
        noise_flat = noise.reshape(B, C, -1)
        squeezed_noise_flat = torch.bmm(squeeze_matrices, noise_flat)
        return squeezed_noise_flat.reshape(B, C, H, W)


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
import torch
import shutil # Import shutil for cleaning up directories

class ConsistentCustomDDPMScheduler(DDPMScheduler):
    def __init__(self, noise_squeezer=None, **kw):
        super().__init__(**kw)
        self.noise_squeezer = noise_squeezer

    # ------------------------------------
    # Forward pass (unchanged)
    # ------------------------------------
    def add_noise(self, x0, eps, t):
        if self.noise_squeezer and self.noise_squeezer.squeeze_strength != 0:
            S   = self.noise_squeezer.get_squeeze_matrix(x0, t, self)
            eps = self.noise_squeezer.apply_squeeze(eps, S)
        xt = super().add_noise(x0, eps, t)       # parent already handles shapes
        return xt, eps                           # eps is the *training target*

    # ------------------------------------
    # Reverse pass
    # ------------------------------------
    def step(self, eps_pred_sq, t, x_t_sq, generator=None, return_dict=True):
        """eps_pred_sq  and  x_t_sq live in *squeezed* coords."""
        if (self.noise_squeezer is None or
            self.noise_squeezer.squeeze_strength == 0):
            # vanilla path
            return super().step(eps_pred_sq, t, x_t_sq,
                                generator=generator, return_dict=return_dict)

        # ---- 1. map to unsqueezed coordinates ---------------------------
        S        = self.noise_squeezer.get_squeeze_matrix(eps_pred_sq, t, self)
        S_inv    = torch.linalg.inv(S)           # fast b/c dim = 3
        x_t      = self.noise_squeezer.apply_squeeze(x_t_sq,  S_inv)
        eps_pred = self.noise_squeezer.apply_squeeze(eps_pred_sq, S_inv)

        # ---- 2. perform the *standard* DDPM update ----------------------
        out      = super().step(eps_pred, t, x_t,
                                generator=generator, return_dict=True)

        # ---- 3. map the result back to squeezed coords ------------------
        prev_sq  = self.noise_squeezer.apply_squeeze(out.prev_sample, S)
        x0_sq    = self.noise_squeezer.apply_squeeze(out.pred_original_sample, S)

        if not return_dict:
            return (prev_sq,)
        return DDPMSchedulerOutput(prev_sample=prev_sq,
                                   pred_original_sample=x0_sq)


# pca_evaluator.py, PCA Analysis on produced images
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from pathlib import Path

@torch.no_grad()
def collect_images(dataloader, n, device):
    imgs = []
    for batch in dataloader:
        imgs.append(batch.to(device))
        if len(imgs)*batch.size(0) >= n:
            break
    imgs = torch.cat(imgs, 0)[:n]        # [n,3,32,32]  in [-1,1]
    return ((imgs + 1) / 2).clamp(0,1)    # → [0,1]

def to_flat(x):           # x: [N,3,32,32] → [N,3072]
    return x.permute(0,2,3,1).reshape(x.size(0), -1).cpu().numpy()

def pca_analysis(real_imgs, gen_imgs, k=50, outdir="pca_plots"):
    Path(outdir).mkdir(exist_ok=True)

    pca = PCA(n_components=k, svd_solver="randomized", whiten=False)
    pca.fit(to_flat(real_imgs))

    Z_real = pca.transform(to_flat(real_imgs))
    Z_gen  = pca.transform(to_flat(gen_imgs))

    # 1)  Explained-variance curves
    evr   = pca.explained_variance_ratio_
    plt.figure(figsize=(5,3.5))
    plt.semilogy(np.arange(1,k+1), evr.cumsum(), 'b-o', label="CIFAR-10")
    evr_gen = np.var(Z_gen, 0) / np.sum(np.var(Z_real,0))   # same denom
    plt.semilogy(np.arange(1,k+1), evr_gen.cumsum(), 'r-o', label="Generated")
    plt.xlabel("num PCs"); plt.ylabel("cumulative explained var.")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/explained_var.png"); plt.close()

    # 2)  PC-plane scatter
    plt.figure(figsize=(4,4))
    plt.scatter(Z_real[:,0], Z_real[:,1], s=8, alpha=.2, label="real")
    plt.scatter(Z_gen[:,0],  Z_gen[:,1],  s=8, alpha=.2, label="gen")
    plt.xlabel("PC-1"); plt.ylabel("PC-2"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/pc1_pc2.png"); plt.close()

    # 3) Simple distance metrics in PC-space
    mu_r, mu_g = Z_real.mean(0), Z_gen.mean(0)
    cov_r      = np.cov(Z_real, rowvar=False)
    cov_g      = np.cov(Z_gen,  rowvar=False)

    # – Fréchet in PC-space (optionally whitened)
    from scipy.linalg import sqrtm
    diff   = mu_r - mu_g
    covavg = sqrtm(cov_r @ cov_g)
    fid_pc = diff@diff + np.trace(cov_r + cov_g - 2*covavg.real)
    return fid_pc


def train_loop(config):
    """Main training loop with noise squeezing support."""

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard" if config["use_tensorboard"] else None,
        project_dir=os.path.join(config["output_dir"], "logs"),
    )

    if accelerator.is_main_process:
        wandb.init(
            project="squeezing",
            name=config.get("run_name", None),
            config=config
        )
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [-1, 1] range
    ])

    dataset = CIFAR10ImagesOnly(
        root=config["dataset_path"],
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    #  **NEW**  EVALUATION dataloader  (NO augmentation, NO randomness)
    # ------------------------------------------------------------------
    transform_eval = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset_eval = CIFAR10ImagesOnly(
        root=config["dataset_path"],
        train=True,              # still training split for CIFAR-10 stats
        transform=transform_eval,
        download=True
    )

    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=config["eval_batch_size"],
        shuffle=False,           # deterministic order
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Initialize noise squeezer and calculate PCA
    noise_squeezer = NoiseSqueezer(
        squeeze_strength=config["squeeze_strength"],
        quantum_limited=config["quantum_limited"]
    )

    if config["squeeze_strength"] != 0:
        noise_squeezer.calculate_dataset_statistics(
            dataloader,
            accelerator.device,
            max_samples=config["pca_samples"]
        )

    # Model
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Scheduler with custom noise support
    noise_scheduler = ConsistentCustomDDPMScheduler(
        noise_squeezer=noise_squeezer,
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"],
        prediction_type=config["prediction_type"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
        eps=config["adam_epsilon"],
    )

    # Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=(len(dataloader) * config["num_epochs"]),
    )

    # Prepare everything with accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # WandB watch model
    if accelerator.is_main_process:
        wandb.watch(accelerator.unwrap_model(model), log=None)

    # Create EMA model if enabled
    ema_model = None
    if config.get("ema_decay", 0) > 0:
          ema_model = EMAModel(
            model.parameters(),
            decay=config["ema_decay"],
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=3/4
          )

          ema_model.to(accelerator.device)
          accelerator.register_for_checkpointing(ema_model)

    # Initialize evaluation calculators based on metric type
    if config["evaluation_metric"] == "fid":
        evaluator = FIDCalculator(device=accelerator.device)
    elif config["evaluation_metric"] == "is":
        evaluator = InceptionScoreCalculator(device=accelerator.device)
    elif config["evaluation_metric"] == "pr":
        evaluator = PrecisionRecallCalculator(device=accelerator.device, k=config["pr_k_neighbors"])
    else:
        raise ValueError(f"Unknown evaluation metric: {config['evaluation_metric']}")

    # Training
    global_step = 0
    for epoch in range(config["num_epochs"]):
        progress_bar = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{config['num_epochs']}"
        )

        for step, batch in enumerate(dataloader):
            clean_images = batch
            # Sample noise
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]

            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # CHANGED: add_noise now returns squeezed noise as target
            noisy_images, noise_target = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise
                model_output = model(noisy_images, timesteps).sample

                # CHANGED: Calculate loss - model now predicts squeezed noise
                if config["prediction_type"] == "epsilon":
                    target = noise_target  # Squeezed noise
                elif config["prediction_type"] == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise_target, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {config['prediction_type']}")

                loss = F.mse_loss(model_output, target)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update EMA model
            # We do this on the main process and only on steps where the optimizer updates the model
            if accelerator.is_main_process:
                if config.get("ema_decay", 0) > 0:
                    ema_model.step(model.parameters())

            progress_bar.update(1)
            logs = {"train/loss": loss.detach().item(), "train/lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # WandB per step log
            if accelerator.is_main_process:
                wandb.log({**logs, "train_step": global_step}, commit=True)

            global_step += 1

        progress_bar.close()

        # Evaluation at specified intervals
        if accelerator.is_main_process and ((epoch + 1) % config["eval_epochs"] == 0 or epoch == 0):

            # --- CHECKPOINTING LOGIC ---
            checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint-epoch-{epoch + 1}")
            accelerator.save_state(checkpoint_dir)
            print(f"Saved training state to {checkpoint_dir}")

            # If EMA is enabled, store the main model's weights and copy over the EMA weights
            if ema_model is not None:
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())

            # --- Perform Evaluation ---
            # The main 'model' now has the EMA weights for evaluation
            print(f"\n[Epoch {epoch+1}] Generating sample images...")
            evaluate_and_save_samples(
                model=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
                config=config,
                epoch=epoch + 1,
                device=accelerator.device
            )

            # Saving denoising trajectories also
            save_denoising_trajectories(
                model=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
                config=config,
                epoch=epoch + 1,
                device=accelerator.device,
            )

            # WandB: log sample grid and denoising GIF if present
            if accelerator.is_main_process:

                grid_path = os.path.join(config["output_dir"], "samples", f"grid_epoch{epoch + 1}.png")
                eval_step = epoch + 1
                if os.path.exists(grid_path):
                    wandb.log({"eval/samples_grid": wandb.Image(grid_path),"eval_step": eval_step})

                gif_path = os.path.join(config["output_dir"], "denoising", f"traj_epoch{epoch + 1}.gif")
                if os.path.exists(gif_path):
                    wandb.log({"eval/denoising_traj": wandb.Video(gif_path),"eval_step": eval_step})


            # Evaluate with chosen metric
            if config["evaluation_metric"] == "fid":
                score = evaluate_fid_score(
                    model=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                    dataloader=dataloader_eval,
                    fid_calculator=evaluator,
                    config=config,
                    epoch=epoch + 1,
                    device=accelerator.device
                )
                accelerator.log({"fid_score": score}, step=epoch)
                if accelerator.is_main_process:
                    wandb.log({"eval/fid": score, "eval_step": eval_step})

            elif config["evaluation_metric"] == "is":
                mean_is, std_is = evaluate_inception_score(
                    model=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                    is_calculator=evaluator,
                    config=config,
                    epoch=epoch + 1,
                    device=accelerator.device
                )
                accelerator.log({"inception_score": mean_is, "is_std": std_is}, step=epoch)
                if accelerator.is_main_process:
                    wandb.log({"eval/is_mean": mean_is, "eval/is_std": std_is, "eval_step": eval_step})

            elif config["evaluation_metric"] == "pr":
                 precision, recall = evaluate_pr_score(
                    model=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                    dataloader_eval=dataloader_eval,
                    pr_calculator=evaluator,
                    config=config,
                    epoch=epoch + 1,
                    device=accelerator.device
                )
                 accelerator.log({"precision": precision, "recall": recall}, step=epoch)
                 if accelerator.is_main_process:
                     wandb.log({"eval/precision": precision, "eval/recall": recall, "eval_step": eval_step})


            # If EMA was used, restore the original training weights
            if ema_model is not None:
                ema_model.restore(model.parameters())

    # Save final model
    if accelerator.is_main_process:

        # If EMA is enabled, copy the EMA weights to the model for the final save
        if ema_model is not None:
            ema_model.copy_to(model.parameters())

        # Now 'model' contains the best (EMA) weights for saving.
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(config["output_dir"])
        wandb.finish()

@torch.no_grad()
def evaluate_and_save_samples(model, scheduler, config, epoch, device):
    """Generate and save sample images."""
    model.eval()

    # Sample some noise
    n_samples = config.get("num_save_samples", 64)
    image_shape = (n_samples, 3, config["image_size"], config["image_size"])

    if scheduler.noise_squeezer and scheduler.noise_squeezer.squeeze_strength != 0:
        # Initialize with squeezed noise
        noise = torch.randn(image_shape, device=device)
        t = torch.full((n_samples,), scheduler.config.num_train_timesteps - 1, device=device)
        squeeze_matrices = scheduler.noise_squeezer.get_squeeze_matrix(noise, t, scheduler)
        image = scheduler.noise_squeezer.apply_squeeze(noise, squeeze_matrices)
    else:
        image = torch.randn(image_shape, device=device)

    # Denoise
    for t in scheduler.timesteps:
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = model(image, t_batch).sample

        # Compute previous image
        image = scheduler.step(noise_pred, t, image).prev_sample

    # Denormalize
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    # Make grid with larger display size
    images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image]

    # Resize images for better visibility
    display_size = config.get("display_size", 128)  # Default 128x128 for display
    images_resized = [img.resize((display_size, display_size), Image.Resampling.NEAREST) for img in images]

    # Calculate grid dimensions
    n_images = len(images_resized)
    cols = int(np.sqrt(n_images))
    rows = (n_images + cols - 1) // cols

    image_grid = make_image_grid(images_resized, rows=rows, cols=cols)

    # Save both the grid and individual samples
    save_dir = os.path.join(config["output_dir"], "samples")
    os.makedirs(save_dir, exist_ok=True)

    # Save grid
    image_grid.save(os.path.join(save_dir, f"grid_epoch{epoch}.png"))

    # Optionally save individual high-res samples
    if config.get("save_individual_samples", False):
        individual_dir = os.path.join(save_dir, f"individuals_step{epoch}")
        os.makedirs(individual_dir, exist_ok=True)
        for i, img in enumerate(images_resized):
            img.save(os.path.join(individual_dir, f"sample_{i}.png"))

    model.train()


# ------------------------------------------------------------
#  Code to save denoising trajectories
# ------------------------------------------------------------
from diffusers.utils import make_image_grid

@torch.no_grad()
def save_denoising_trajectories(
    model,
    scheduler,
    config,
    epoch: int,
    device,
):
    """
    Generate a handful of denoising *trajectories* (noise → image) and save:
      • a PNG grid per sample  (frames left→right = early→late)
      • one animated GIF that shows all samples in lock-step
    """
    model.eval()

    n_samples = config.get("num_denoising_samples", 4)          # ≤16 is comfortable
    n_frames  = config.get("num_denoising_frames", 8)           # frames per sample
    # Choose ~uniformly-spaced timesteps to snapshot
    capture_ids = set(
        np.linspace(0, len(scheduler.timesteps) - 1, n_frames, dtype=int).tolist()
    )

    shape = (n_samples, 3, config["image_size"], config["image_size"])
    # -------- initial noise (respect squeezing if enabled) -------
    if scheduler.noise_squeezer and scheduler.noise_squeezer.squeeze_strength != 0:
        noise = torch.randn(shape, device=device)
        t_max = torch.full((n_samples,), scheduler.config.num_train_timesteps - 1, device=device)
        S     = scheduler.noise_squeezer.get_squeeze_matrix(noise, t_max, scheduler)
        img   = scheduler.noise_squeezer.apply_squeeze(noise, S)
    else:
        img = torch.randn(shape, device=device)

    # We will store *all* frames for the GIF, but only `capture_ids`
    # for the per-sample grids
    gif_frames     = []
    captured_grids = [[] for _ in range(n_samples)]

    for idx, t in enumerate(scheduler.timesteps):
        t_batch   = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps       = model(img, t_batch).sample
        img       = scheduler.step(eps, t, img).prev_sample

        # --- store current frame ---------------------------------
        img_vis = ((img / 2 + 0.5).clamp(0, 1)
                   .cpu().permute(0, 2, 3, 1).numpy())  # [N,H,W,C], float32

        gif_frames.append((img_vis * 255).astype(np.uint8))

        if idx in capture_ids:
            for k in range(n_samples):
                captured_grids[k].append(
                    Image.fromarray((img_vis[k] * 255).astype(np.uint8))
                )

    # -------------------------------------------------------------
    #  WRITE FILES
    # -------------------------------------------------------------
    out_dir = os.path.join(config["output_dir"], "denoising")
    os.makedirs(out_dir, exist_ok=True)

    # (1) per-sample PNG grids
    for k, frame_list in enumerate(captured_grids):
        grid = make_image_grid(frame_list, rows=1, cols=len(frame_list))
        grid.save(os.path.join(out_dir, f"traj_epoch{epoch}_sample{k}.png"))

    # (2) global GIF (all samples tiled  in a grid)
    #     ––> shape: T × H × W × C ; we tile N samples into √N × √N grid
    frames = []
    # convert every frame to PIL after tiling
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    for f in gif_frames:
        # f : [N,H,W,C] uint8
        tiles   = [Image.fromarray(fi) for fi in f]
        big     = make_image_grid(tiles, rows=n_rows, cols=n_cols)
        frames.append(big)

    gif_path = os.path.join(out_dir, f"traj_epoch{epoch}.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,   # ms per frame
        loop=0,
    )
    print(f"[Epoch {epoch}]  saved denoising trajectories → {gif_path}")

    model.train()

# Configuration
SQUEEZE_STRENGTH = -0.4 # 0.0 = standard diffusion, higher = more squeezing
config = {
    # Data
    "dataset_path": "./cifar10_data",
    "image_size": 32,
    "train_batch_size": 128,
    "eval_batch_size": 1024,
    "num_workers": 4,

    # Model
    "num_train_timesteps": 1000,
    "beta_schedule": "linear",
    "prediction_type": "epsilon",  # or "v_prediction"

    # Noise squeezing
    "squeeze_strength": SQUEEZE_STRENGTH,
    "quantum_limited": False,  # Use volume-preserving squeeze
    "pca_samples": 50000,  # Number of samples for PCA calculation

    # Training
    "num_epochs":  30,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "lr_warmup_steps": 500,
    "adam_beta1": 0.95,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-6,
    "adam_epsilon": 1e-8,
    "ema_decay":  0.9999,  # EMA decay rate (0 = disabled)

    # Logging
    "mixed_precision": "fp16",
    "output_dir": "./ddpm_10kFID_s" + str(SQUEEZE_STRENGTH) +"_EMA_PyTorchInception",
    "run_name": "ddpm_10kFID_s" + str(SQUEEZE_STRENGTH),
    "use_tensorboard": True,
    'num_save_samples': 36, # Number of images generated
    # Denoising trajectories
    "num_denoising_samples": 4,   # how many trajectories to save
    "num_denoising_frames": 8,    # snapshots per trajectory

    # Display settings
    "display_size": 96,  # Size to display samples (they're still 32x32 internally)
    "save_individual_samples": False,  # Set True to also save individual images

    # Evaluation settings
    "evaluation_metric": "fid",  # "fid", "is" or "pr"
    "eval_epochs": 5,  # Calculate metric/generate images/save model every N epochs

    # FID specific settings
    "fid_n_real": 10000,  # Number of real samples for FID
    "fid_n_generated": 10000,  # Number of generated samples for FID
    "fid_batch_size": 50,  # Batch size for FID feature extraction
    "fid_inference_steps": 50,  # Number of denoising steps for FID evaluation

    # IS specific settings
    "is_n_generated": 10000,  # Number of generated samples for IS
    "is_batch_size": 1024,  # Batch size for IS calculation
    "is_inference_steps": 50,  # Number of denoising steps for IS evaluation
    "is_splits": 10,  # Number of splits for IS calculation

    # PR specific settings
    "pr_n_real": 50000, # Number of real samples for PR
    "pr_n_generated": 10000, # Number of generated samples for PR
    "pr_batch_size": 50, # Batch size for PR feature extraction - NOTE: Not used by torch-fidelity directly for data loading, but kept for consistency
    "pr_inference_steps": 50, # Number of denoising steps for PR evaluation
    "pr_k_neighbors": 3, # Number of neighbors for k-NN in PR calculation


    # Seed
    "seed": 42,
}


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Train
    train_loop(config)