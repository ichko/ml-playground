# @title Utilities

import kornia
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import requests
import PIL
import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import Image, HTML, clear_output, display
from ipywidgets import Output
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from matplotlib import cm


def load_image(url, max_size=100):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img_np = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img_np[..., :3] *= img_np[..., 3:]
    return img_np, img


def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    return load_image(url)


def plot_loss(loss):
    plt.figure(figsize=(10, 4))
    plt.title("Loss history (log10)")
    plt.plot(loss, ".", alpha=0.2)
    plt.show()


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)

        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


class EasyTensor:
    def __init__(self, tensor):
        self.raw = tensor

    def sq(self, dim=0):
        return EasyTensor(self.raw.squeeze(dim=dim))

    def usq(self, dim=0):
        return EasyTensor(self.raw.unsqueeze(dim=dim))

    @property
    def chw(self):
        return EasyTensor(self.raw.permute(0, 3, 1, 2))

    @property
    def chw(self):
        return EasyTensor(self.raw.permute(0, 3, 1, 2))

    @property
    def hwc(self):
        return EasyTensor(self.raw.permute(0, 2, 3, 1))

    @property
    def np(self):
        return self.raw.detach().cpu().numpy()

    def imshow(self):
        plt.imshow(self.np)


def register_easy():
    torch.Tensor.ez = property(lambda self: EasyTensor(self))


class CAEncoder(nn.Module):
    def __init__(self, num_chanels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=num_chanels,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_chanels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

        # self.net[0].weight.data.normal_()
        self.net[0].bias.data.zero_()
        # self.net[2].weight.data.normal_()
        self.net[2].bias.data.zero_()
        # self.net[4].weight.data.normal_()
        self.net[4].bias.data.zero_()

    def forward(self, x, steps):
        seq = [x]
        for i in range(steps):
            x = x + self.net(x)
            seq.append(x)
        return torch.stack(seq, dim=1)


class Decoder(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        return self.net(x)


class NoisyChannel(nn.Module):
    def __init__(
        self,
        msg_size,
        seed_shape,
        encoder,
        decoder,
        noise,
        num_decode_channels,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise
        self.num_decode_channels = num_decode_channels

        self.msg_size = msg_size
        self.seed_shape = seed_shape
        self.optim = torch.optim.Adam(
            [*self.encoder.parameters(), *self.decoder.parameters()],
            lr=0.001,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def generate_msg(self, bs):
        return torch.randn(bs, self.msg_size, device=self.device)

    def generate_seeded_input(self, bs):
        msg = self.generate_msg(bs)
        c, h, w = self.seed_shape
        x = torch.zeros(*[bs, *self.seed_shape], device=self.device)
        mid_w, mid_h = w // 2, h // 2
        x[:, -self.msg_size :, mid_h, mid_w] = msg
        return x, msg

    def forward(self, bs, steps):
        input, msg = self.generate_seeded_input(bs)
        x = self.encoder(input, steps)
        return x[:, :, : self.num_decode_channels]

    def training_step(self, bs, steps):
        self.train()
        self.optim.zero_grad()

        input, msg = self.generate_seeded_input(bs)
        x = self.encoder(input, steps // 2)[:, -1]
        x = self.noise(x)
        x = self.encoder(x, steps // 2)[:, -1]

        msg_pred = self.decoder(x[:, : self.num_decode_channels])

        loss = F.mse_loss(msg_pred, msg)

        loss.backward()
        self.optim.step()

        return loss.item()


def generate_video(model, steps=250):
    model.eval()
    with torch.no_grad():
        seq = model(bs=10, steps=steps)
        is_single_channel = seq.shape[2] == 1  # if image num channels is 1

        # TODO: This should be vectorized
        grid_video = []
        for i in range(seq.size(1)):  # for each time step
            snap = seq[:, i]
            grid = torchvision.utils.make_grid(snap, nrow=5, padding=1)

            if is_single_channel:
                grid = grid[:1]
            grid_video.append(grid)

        grid_video = torch.stack(grid_video, dim=0)
        grid_video = grid_video.permute(0, 2, 3, 1)

        with VideoWriter("test.ignore.mp4") as vid:
            for frame in grid_video:
                np_frame = frame.ez.np

                if np_frame.shape[-1] == 1:
                    np_frame = cm.viridis(np_frame[:, :, 0])[:, :, :3]

                np_frame = zoom(np_frame, scale=5)
                vid.add(np_frame)

    nonce = np.random.rand()

    return HTML(
        f"""<video controls autoplay loop>
                <source src="test.ignore.mp4#{nonce}" type="video/mp4">
            </video>
        """
    )
