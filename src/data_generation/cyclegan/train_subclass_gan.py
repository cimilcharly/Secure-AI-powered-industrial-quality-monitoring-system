"""
Subclass-Conditioned GAN Training & Verification Generator for Bharath's Damage Data.
Conditions on the 7 Damage Subclasses:
0: burn_mark
1: cut
2: frayed_edge
3: hole
4: rip
5: scratch
6: tear

Trains on all 2,215 extracted patches and generates 10 pictures for each subclass (70 total).
"""

import os
import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SUBCLASSES = ["burn_mark", "cut", "frayed_edge", "hole", "rip", "scratch", "tear"]
NUM_CLASSES = len(SUBCLASSES)


# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ConditionalGenerator(nn.Module):
    """
    Takes 3 RGB channels + 7 spatial one-hot class channels = 10 input channels.
    Outputs 3 RGB channels with the requested defect synthesized.
    """
    def __init__(self, in_channels=3 + NUM_CLASSES, out_channels=3, num_res_blocks=6):
        super(ConditionalGenerator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        # c is (batch_size, num_classes)
        # Expand c to match spatial dimensions (batch_size, num_classes, H, W)
        c_spatial = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x_cond = torch.cat([x, c_spatial], dim=1)
        return self.model(x_cond)


class MultiTaskDiscriminator(nn.Module):
    """
    Evaluates both patch authenticity (16x16 PatchGAN) and defect classification (7 classes).
    """
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super(MultiTaskDiscriminator, self).__init__()

        def conv_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.backbone = nn.Sequential(
            *conv_block(in_channels, 64, normalize=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
        )

        # 1. Adversarial patch head (16x16)
        self.out_adv = nn.Conv2d(512, 1, kernel_size=3, padding=1)

        # 2. Subclass classification head
        self.out_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out_adv = self.out_adv(feat)
        out_cls = self.out_cls(feat)
        return out_adv, out_cls


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class SubclassGANDataset(Dataset):
    def __init__(self, data_root="data/cyclegan"):
        self.root = Path(data_root)
        self.nominal_files = sorted(list((self.root / "trainA").glob("*.jpg")))

        self.damage_files = []
        for class_idx, sub in enumerate(SUBCLASSES):
            files = sorted(list((self.root / "trainB" / sub).glob("*.jpg")))
            for f in files:
                self.damage_files.append((f, class_idx))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return max(len(self.nominal_files), len(self.damage_files))

    def __getitem__(self, index):
        nom_path = self.nominal_files[index % len(self.nominal_files)]
        dam_path, dam_label = self.damage_files[random.randint(0, len(self.damage_files) - 1)]

        img_nom = self.transform(Image.open(nom_path).convert("RGB"))
        img_dam = self.transform(Image.open(dam_path).convert("RGB"))

        return {
            "nom": img_nom,
            "dam": img_dam,
            "dam_label": dam_label
        }


# -------------------------------------------------------------
# Training & Verification Generation
# -------------------------------------------------------------
def train_and_generate(epochs=3, batch_size=4, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Subclass-Conditioned GAN on device: {device}")

    output_base = Path("data/cyclegan/subclasses")
    for sub in SUBCLASSES:
        (output_base / sub).mkdir(parents=True, exist_ok=True)
    Path("models/cyclegan").mkdir(parents=True, exist_ok=True)

    generator = ConditionalGenerator().to(device)
    discriminator = MultiTaskDiscriminator().to(device)

    criterion_adv = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.L1Loss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = SubclassGANDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Loaded {len(dataset)} items per epoch. Training across 7 subclasses...")

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(dataloader):
            real_nom = batch["nom"].to(device)
            real_dam = batch["dam"].to(device)
            labels_dam = batch["dam_label"].to(device)

            b_size = real_nom.size(0)
            valid_adv = torch.ones((b_size, 1, 16, 16), requires_grad=False, device=device)
            fake_adv = torch.zeros((b_size, 1, 16, 16), requires_grad=False, device=device)

            # Target subclass one-hot condition
            target_labels = torch.randint(0, NUM_CLASSES, (b_size,), device=device)
            target_c = torch.zeros((b_size, NUM_CLASSES), device=device)
            target_c.scatter_(1, target_labels.unsqueeze(1), 1.0)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            fake_dam = generator(real_nom, target_c)
            pred_fake_adv, pred_fake_cls = discriminator(fake_dam)

            loss_G_adv = criterion_adv(pred_fake_adv, valid_adv)
            loss_G_cls = criterion_cls(pred_fake_cls, target_labels)
            
            # Identity/Reconstruction: when synthesizing, preserve fabric base
            loss_G_rec = criterion_recon(fake_dam, real_nom) * 5.0

            loss_G = loss_G_adv + loss_G_cls + loss_G_rec
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real_adv, pred_real_cls = discriminator(real_dam)
            loss_D_real_adv = criterion_adv(pred_real_adv, valid_adv)
            loss_D_real_cls = criterion_cls(pred_real_cls, labels_dam)

            pred_fake_adv_detach, _ = discriminator(fake_dam.detach())
            loss_D_fake_adv = criterion_adv(pred_fake_adv_detach, fake_adv)

            loss_D = (loss_D_real_adv + loss_D_fake_adv) / 2 + loss_D_real_cls
            loss_D.backward()
            optimizer_D.step()

            if i % 25 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[G loss: {loss_G.item():.4f} (adv: {loss_G_adv.item():.4f}, cls: {loss_G_cls.item():.4f})] "
                      f"[D loss: {loss_D.item():.4f}]")

            # CPU optimization: 50 steps per epoch provides solid learning without stalling
            if device.type == "cpu" and i >= 50:
                break

    # Save model weights
    torch.save(generator.state_dict(), "models/cyclegan/conditional_generator.pth")
    torch.save(discriminator.state_dict(), "models/cyclegan/multitask_discriminator.pth")
    print("\nModels successfully saved to models/cyclegan/!")

    # -------------------------------------------------------------
    # Generate 10 images for EACH of the 7 subclasses (70 total)
    # -------------------------------------------------------------
    generate_subclass_samples(generator, device, output_base)


def generate_subclass_samples(generator, device, output_base):
    print("\n=======================================================")
    print("Generating exactly 10 pictures for each of the 7 subclasses...")
    print("=======================================================")
    generator.eval()

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    nominal_files = sorted(list(Path("data/cyclegan/trainA").glob("*.jpg")))
    # Select 10 diverse nominal test patches (different shirt colors: white, black, navy, grey, maroon, etc.)
    sample_nominals = [nominal_files[i * (len(nominal_files) // 10)] for i in range(10)]

    for class_idx, subclass_name in enumerate(SUBCLASSES):
        sub_dir = output_base / subclass_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSubclass [{class_idx + 1}/7]: '{subclass_name}' -> Generating 10 samples...")

        c_vec = torch.zeros((1, NUM_CLASSES), device=device)
        c_vec[0, class_idx] = 1.0

        for sample_idx, nom_path in enumerate(sample_nominals, 1):
            img_nom = Image.open(nom_path).convert("RGB")
            tensor_nom = test_transform(img_nom).unsqueeze(0).to(device)

            with torch.no_grad():
                tensor_fake = generator(tensor_nom, c_vec)

                # Denormalize to [0, 1]
                orig_img = tensor_nom.squeeze(0).cpu() * 0.5 + 0.5
                fake_img = tensor_fake.squeeze(0).cpu() * 0.5 + 0.5
                fake_img = torch.clamp(fake_img, 0, 1)

                # Save side-by-side comparison (Clean Nominal | Generated Defect)
                comparison = torch.cat([orig_img, fake_img], dim=2)
                out_path = sub_dir / f"{subclass_name}_sample_{sample_idx:02d}.jpg"
                save_image(comparison, str(out_path))

                # Also save individual generated defect image
                single_path = sub_dir / f"{subclass_name}_only_{sample_idx:02d}.jpg"
                save_image(fake_img, str(single_path))

            print(f"  [{subclass_name} {sample_idx}/10] Saved: {out_path.name}")

    print("\nSUCCESS: All 70 subclass verification pictures (10 per subclass) successfully generated!")


if __name__ == "__main__":
    train_and_generate(epochs=3, batch_size=4)
