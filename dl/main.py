import os
from argparse import Namespace

from utils import Lambda, SpatialLinearTransformer, Module, SpatialUVTransformer

from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import cv2

import wandb


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, repeat=1, device="cpu"):
        super().__init__()
        self.repeat = repeat
        self.device = device
        self.imgs = self.load_images(".data/x")
        self.target = self.load_images(".data/y")[0]

    def __len__(self):
        return len(self.imgs) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.imgs)
        return {
            "x": self.imgs[idx].to(self.device),
            "y": self.target.to(self.device),
        }

    @staticmethod
    def get_example_batch(size, device):
        dl = DocumentDataset.get_dl(
            bs=size,
            shuffle=False,
            repeat=size,
            device=device,
        )
        it = iter(dl)
        batch = next(it)
        return batch

    @staticmethod
    def load_images(path):
        file_names = [f for f in os.listdir(path) if f.endswith("jpg")]
        imgs = []
        scale = 0.1
        W, H = 2138, 2997
        w, h = int(W * scale), int(H * scale)

        for file_name in file_names:
            im_path = os.path.join(path, file_name)
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            i_h, i_w, _ = img.shape
            if i_w > i_h:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (w, h))
            img = img.transpose(2, 0, 1)
            imgs.append(img / 255)

        return torch.tensor(imgs, dtype=torch.float32)

    @staticmethod
    def get_dl(bs, shuffle, repeat=1, device="cpu"):
        dataset = DocumentDataset(
            repeat=repeat,
            device=device,
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=bs,
            shuffle=shuffle,
        )
        return dl

    @staticmethod
    def sanity_check():
        import matplotlib.pyplot as plt

        dl = DocumentDataset.get_dl(bs=16, shuffle=False, repeat=10)
        it = iter(dl)
        batch = next(it)
        batch["x"].wrap.grid(nr=4).imshow(figsize=(8, 8))
        # plt.close()


class GeometricTransformModel(Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        # self.st = SpatialUVTransformer(
        #     i=1000,
        #     uv_resolution_shape=(30, 20),
        # )
        self.st = SpatialLinearTransformer(
            i=1000,
            num_channels=1,
        )

    def forward(self, x):
        self.features = self.feature_extractor(x)
        x = x.mean(dim=1, keepdim=True)
        y_hat = self.st([self.features, x])
        return y_hat

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        return F.binary_cross_entropy(y_hat, y)

    def optim_step(self, optim, batch):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        if loss.requires_grad:
            optim.zero_grad()
            loss.backward()
            optim.step()

        return {
            "loss": loss.item(),
            "features": self.features,
            "y_hat": y_hat,
        }


def main():
    DEVICE = "cuda"
    example_batch = DocumentDataset.get_example_batch(
        size=16,
        device=DEVICE,
    )
    dl = DocumentDataset.get_dl(
        bs=32,
        shuffle=False,
        repeat=100,
        device=DEVICE,
    )
    model = GeometricTransformModel()
    hparams = Namespace(lr=0.00003)
    optim = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    epochs = 200

    model = model.to(DEVICE)

    wandb.init(
        name="UV Transformer",
        dir=".reports",
        project="rectify",
        config=dict(
            vars(hparams),
            name=model.name,
            model_num_params=model.count_parameters(),
        ),
    )

    epoch_bar = tqdm(range(epochs))
    for _e in epoch_bar:
        batch_bar = tqdm(dl)
        model.train()
        for batch in batch_bar:
            optim_info = model.optim_step(optim, batch)
            loss = optim_info["loss"]
            wandb.log({"train_loss": loss})

            batch_bar.set_description(f"Loss: {loss:.5f}")

        with torch.no_grad():
            model.eval()
            example_info = model.optim_step(optim, example_batch)
        imgs = example_info["y_hat"]
        wandb.log({"example_batch_imgs": [wandb.Image(i) for i in imgs]})


if __name__ == "__main__":
    main()
