import os

import torch
import torchvision as tv
import cv2

import utils


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.imgs = self.load_images(path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

    @staticmethod
    def load_images(path):
        file_names = [f for f in os.listdir(path) if f.endswith('jpg')]
        imgs = []
        scale = 0.2
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
    def get_dl(bs, shuffle):
        dataset = DocumentDataset(path='.data')
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bs, shuffle=shuffle)
        return dl

    @staticmethod
    def example():
        import matplotlib.pyplot as plt

        dl = DocumentDataset.get_dl(bs=8, shuffle=False)
        it = iter(dl)
        batch = next(it)
        batch.spread[4, 2, -3:].imshow(figsize=(16, 16))

        # plt.imshow(img)
