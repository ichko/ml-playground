import logging
import os
from argparse import Namespace

import cv2
import ez_torch
import kornia
import torch
import torchvision
import wandb
from ez_torch.models import Module, SpatialUVOffsetTransformer
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

logger = logging.getLogger()


class DocumentDataset(torch.utils.data.Dataset):
    W = 2138
    H = 2997
    scale = 0.1
    w, h = int(W * scale), int(H * scale)
    ar = W / H

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

    @classmethod
    def load_images(cls, path):
        file_names = [f for f in os.listdir(path) if f.endswith("jpg")]
        imgs = []

        for file_name in file_names:
            im_path = os.path.join(path, file_name)
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            i_h, i_w, _ = img.shape
            if i_w > i_h:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (cls.w, cls.h))
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
        batch["x"].ez.grid(nr=4).imshow(figsize=(8, 8))
        # plt.close()


class GeometricTransformModel(Module):
    def __init__(self, res_w, res_h):
        super().__init__()

        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.st = SpatialUVOffsetTransformer(
            i=1000,
            uv_resolution_shape=(res_w, res_h),
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
    hparams = Namespace(lr=0.00003, epochs=10)
    wandb.init(
        name="UV Transformer",
        dir=".reports",
        project="rectify",
        config=hparams,
    )

    for res in range(5, 10):
        res_w = res
        res_h = int(res_w / DocumentDataset.ar)
        logging.info(f"res {res_w} {res_h}")

        model = GeometricTransformModel(res_w, res_h)
        model = model.to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=hparams.lr)

        for _e in tqdm(range(hparams.epochs)):
            model.train()

            batch_bar = tqdm(dl)
            for batch in batch_bar:
                optim_info = model.optim_step(optim, batch)
                loss = optim_info["loss"]
                wandb.log({"train_loss": loss})

                batch_bar.set_description(f"Loss: {loss:.5f}")

            with torch.no_grad():
                model.eval()
                example_info = model.optim_step(optim, example_batch)
            imgs = example_info["y_hat"].ez.grid(nr=4).raw
            wandb.log({"example_batch_imgs": [wandb.Image(imgs)]})


if __name__ == "__main__":
    main()
