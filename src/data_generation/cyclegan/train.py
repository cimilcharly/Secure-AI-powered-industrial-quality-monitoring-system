"""
CycleGAN Training Pipeline for Bharath's Damage Defect Generation.
Translates:
- Domain A (Nominal Pristine Shirts) <---> Domain B (Damaged Shirts: burns, tears, holes, cuts, frays)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import itertools
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

from src.data_generation.cyclegan.models import GeneratorResNet, DiscriminatorPatchGAN
from src.data_generation.cyclegan.dataset import CycleGANDataset

class ReplayBuffer:
    """Stores previously generated images to stabilize discriminator training."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def train_cyclegan(
    epochs=5,
    batch_size=4,
    lr=0.0002,
    lambda_cyc=10.0,
    lambda_id=5.0,
    checkpoint_dir="models/cyclegan",
    results_dir="data/cyclegan/results"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CycleGAN Training initialized on device: {device}")

    # Output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "A_to_B_damage"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "B_to_A_nominal"), exist_ok=True)

    # Networks
    G_AB = GeneratorResNet(num_residual_blocks=6).to(device)  # Nominal -> Damage
    G_BA = GeneratorResNet(num_residual_blocks=6).to(device)  # Damage -> Nominal
    D_A = DiscriminatorPatchGAN().to(device)
    D_B = DiscriminatorPatchGAN().to(device)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # Replay buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset & DataLoader
    dataset = CycleGANDataset("data/cyclegan", mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Dataset loaded: {len(dataset)} samples per epoch. Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Ground truth labels for PatchGAN (16x16 output)
            valid = torch.ones((real_A.size(0), 1, 16, 16), requires_grad=False, device=device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), requires_grad=False, device=device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss (G_BA(A) should be A, G_AB(B) should be B)
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total Generator loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator A
            # ---------------------
            optimizer_D_A.zero_grad()
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            fake_A_buffered = fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = criterion_GAN(D_A(fake_A_buffered.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------------------
            #  Train Discriminator B
            # ---------------------
            optimizer_D_B.zero_grad()
            loss_real_B = criterion_GAN(D_B(real_B), valid)
            fake_B_buffered = fake_B_buffer.push_and_pop(fake_B)
            loss_fake_B = criterion_GAN(D_B(fake_B_buffered.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            if i % 25 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D_A.item() + loss_D_B.item():.4f}] "
                      f"[G loss: {loss_G.item():.4f} (adv: {loss_GAN.item():.4f}, cyc: {loss_cycle.item():.4f}, id: {loss_identity.item():.4f})]")

            # On CPU, run a fast subset per epoch so training is responsive and finishes promptly
            if device.type == "cpu" and i >= 50:
                break

    # Save generator checkpoints
    torch.save(G_AB.state_dict(), os.path.join(checkpoint_dir, "G_AB_damage.pth"))
    torch.save(G_BA.state_dict(), os.path.join(checkpoint_dir, "G_BA_nominal.pth"))
    print(f"\nModels successfully saved to {checkpoint_dir}!")

    # -------------------------------------------------------------
    # Generate 10 pictures of each domain as requested by the user
    # -------------------------------------------------------------
    generate_verification_samples(G_AB, G_BA, device, results_dir)


def generate_verification_samples(G_AB, G_BA, device, results_dir):
    print("\nGenerating 10 verification pictures for each domain...")
    G_AB.eval()
    G_BA.eval()

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_A_files = sorted(list(Path("data/cyclegan/testA").glob("*.jpg")))[:10]
    test_B_files = sorted(list(Path("data/cyclegan/testB").glob("*.jpg")))[:10]

    # 1. 10 Images: Domain A (Nominal) -> Domain B (Synthesized Damage)
    print("Generating 10 Domain A -> Domain B (Damage) translations...")
    for idx, fpath in enumerate(test_A_files, 1):
        img = Image.open(fpath).convert("RGB")
        tensor_A = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_B = G_AB(tensor_A)
            # Denormalize from [-1, 1] to [0, 1]
            out_img = fake_B.squeeze(0).cpu() * 0.5 + 0.5
            out_img = torch.clamp(out_img, 0, 1)

            # Side-by-side comparison (Original Nominal | Generated Damage)
            orig_img = tensor_A.squeeze(0).cpu() * 0.5 + 0.5
            comparison = torch.cat([orig_img, out_img], dim=2)
            
            out_path = os.path.join(results_dir, "A_to_B_damage", f"sample_{idx:02d}_damage_synth.jpg")
            save_image(comparison, out_path)
            print(f"  [A->B {idx}/10] Saved: {out_path}")

    # 2. 10 Images: Domain B (Damage) -> Domain A (Reconstructed Clean Nominal)
    print("Generating 10 Domain B -> Domain A (Defect Removal) translations...")
    for idx, fpath in enumerate(test_B_files, 1):
        img = Image.open(fpath).convert("RGB")
        tensor_B = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_A = G_BA(tensor_B)
            out_img = fake_A.squeeze(0).cpu() * 0.5 + 0.5
            out_img = torch.clamp(out_img, 0, 1)

            # Side-by-side comparison (Original Damaged | Reconstructed Clean)
            orig_img = tensor_B.squeeze(0).cpu() * 0.5 + 0.5
            comparison = torch.cat([orig_img, out_img], dim=2)

            out_path = os.path.join(results_dir, "B_to_A_nominal", f"sample_{idx:02d}_nominal_recon.jpg")
            save_image(comparison, out_path)
            print(f"  [B->A {idx}/10] Saved: {out_path}")

    print("\nAll 20 verification images (10 per domain) successfully generated!")


if __name__ == "__main__":
    train_cyclegan(epochs=3, batch_size=4)
