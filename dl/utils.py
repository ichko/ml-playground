import os
import math
from functools import wraps

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm


def extend(type):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        setattr(type, func.__name__, func)
        return wrapper

    return decorator


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def count_parameters(self):
        return count_parameters(self)

    def make_persisted(self, path):
        self.path = path

    def persist(self):
        torch.save(self.state_dict(), self.path)

    def preload_weights(self):
        self.load_state_dict(torch.load(self.path))

    def save(self, path=None):
        path = path if self.path is None else path
        torch.save(self, f'{self.path}_whole.h5')

    def can_be_preloaded(self):
        return os.path.isfile(self.path)

    def configure_optim(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def metrics(self, _loss, _info):
        return {}

    def set_requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def summary(self, input_size=-1):
        try:
            from torchsummary import summary
            summary(self, input_size)
            return
        except Exception:
            pass

        result = f' > {self.name[:38]:<38} | {count_parameters(self):09,}\n'
        for name, module in self.named_children():
            type = module._get_name()
            num_prams = count_parameters(module)
            result += f' >  {name[:20]:>20}: {type[:15]:<15} | {num_prams:9,}\n'

        print(result)

    def optim_forward(self, X):
        return self.forward(X)

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def optim_step(self, batch):
        X, y = batch

        y_pred = self.optim_forward(X)
        loss = self.criterion(y_pred, y)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        metrics = self.metrics({
            'loss': loss,
            'X': X,
            'y': y,
            'y_pred': y_pred,
        })

        return {
            'metrics': metrics,
            'loss': loss,
            'y_pred': y_pred,
        }


def pad_in_dim(tensor, pad_size, dim, val=0):
    shape = list(tensor.shape)
    shape[dim] = pad_size - shape[dim]
    padding = torch.full(shape, val).to(tensor.device)
    out = torch.cat([tensor, padding], dim=dim)

    return out


# Trim and/or slice tensor
def fix_dim_size(tensor, size, dim, pad_value=0):
    # Slice only dim
    indices = {dim: slice(0, size)}
    idx = [indices.get(dim, slice(None)) for dim in range(tensor.ndim)]

    tensor = tensor[idx]
    tensor = pad_in_dim(tensor, size, dim, pad_value)

    return tensor


def soft_addressing(keys, address_space, bank):
    """Lets you address tensors from the bank addressed by address_space, using
    the keys.
    Types:
                 keys: Tensor[*A, address_size]
        address_space: Tensor[num_addresses, address_size]
                 bank: Tensor[num_addresses, *B]
               return: Tensor[*A, *B]
    """
    A = keys.size()[:-1]
    address_size = keys.size(-1)

    flat_keys = keys.reshape(-1, address_size)
    selectors = torch.matmul(flat_keys, address_space.T)
    selectors = torch.softmax(selectors, dim=-1)
    selectors = selectors.view(*A, -1)

    return generic_matmul(selectors, bank)


def generic_matmul(first, second):
    """Matmul in the last dim.
    Types:
         first: Tensor[*A,  N]
        second: Tensor[ N, *B]
        return: Tensor[*A, *B]
    """
    A = first.size()[:-1]
    B = second.size()[1:]
    N = first.size(-1)

    flat_first = first.reshape(-1, N)
    flat_second = second.reshape(N, -1)

    return torch.matmul(flat_first, flat_second).view(*A, *B)


def leaky(slope=0.2):
    return nn.LeakyReLU(slope, inplace=True)


def unsqueeze_expand(tensor, dim, times):
    if times == 0:
        return tensor

    tensor = tensor.unsqueeze(dim)
    new_shape = list(tensor.shape)
    new_shape[dim] = times
    return tensor.expand(new_shape)


def reshape_in_time(tensor):
    return tensor.reshape(-1, *tensor.shape[2:])


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def batch_conv(x, w, b, p=0, s=1):
    # SRC - https://discuss.pytorch.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2

    batch_size = x.size(0)
    output_size = w.size(1)

    o = F.conv2d(
        x.reshape(1, batch_size * x.size(1), x.size(2), x.size(3)),
        w.reshape(batch_size * w.size(1), w.size(2), w.size(3), w.size(4)),
        b.reshape(batch_size * b.size(1)),
        groups=batch_size,
        padding=p,
        stride=s,
        dilation=1,
    )
    o = o.reshape(batch_size, output_size, o.size(2), o.size(3))

    return o


def dense(i, o, a=leaky()):
    l = nn.Linear(i, o)
    return l if a is None else nn.Sequential(l, a)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Lambda(nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward

    def forward(self, *args):
        return self.forward(*args)


def resize(t, size):
    return F.interpolate(t, size, mode='bicubic', align_corners=True)


def conv_block(i, o, ks, s, p, a=leaky(), d=1, bn=True):
    block = [nn.Conv2d(i, o, kernel_size=ks, stride=s, padding=p, dilation=d)]
    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    return nn.Sequential(*block)


def deconv_block(i, o, ks, s, p, a=leaky(), d=1, bn=True):
    block = [
        nn.ConvTranspose2d(
            i,
            o,
            kernel_size=ks,
            stride=s,
            padding=p,
            dilation=d,
        )
    ]

    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    if len(block) == 1:
        return block[0]

    return nn.Sequential(*block)


def compute_output_shape(net, frame_shape):
    with torch.no_grad():
        t = torch.rand(1, *frame_shape)
        out = net(t)

    return out.shape


class SpatialTransformer(nn.Module):
    def __init__(self, i, num_channels, only_translations=False):
        super().__init__()

        self.only_translations = only_translations
        self.num_channels = num_channels
        self.locator = nn.Sequential(
            nn.Linear(i, num_channels * 2 * 3),
            Reshape(-1, 2, 3),
        )

        self.device = self.locator[0].bias.device
        # Taken from the pytorch spatial transformer tutorial.
        self.locator[0].weight.data.zero_()
        self.locator[0].bias.data.copy_(
            torch.tensor(
                [1, 0, 0, 0, 1, 0] * num_channels,
                dtype=torch.float,
            ).to(self.device))

    def forward(self, x):
        inp, tensor_3d = x

        theta = self.locator(inp)
        _, C, H, W, = tensor_3d.shape

        if self.only_translations:
            theta[:, :, :-1] = torch.tensor(
                [[1, 0], [0, 1]],
                dtype=torch.float,
            ).to(self.device).unsqueeze_(0)

        grid = F.affine_grid(
            theta,
            (theta.size(dim=0), 1, H, W),
            align_corners=True,
        )

        # Last values
        self.grid = grid
        self.theta = theta

        tensor_3d = tensor_3d.reshape(-1, 1, H, W)
        tensor_3d = F.grid_sample(
            tensor_3d,
            grid,
            align_corners=True,
        )

        return tensor_3d.reshape(-1, C, H, W)


def one_hot(tensor, num_classes, dim):
    tensor = F.one_hot(tensor.long(), num_classes=num_classes).float()
    dim_permutation = list(range(tensor.dim()))
    last_dim = dim_permutation.pop()

    # Place last dim (one-hot) on the desired position
    dim_permutation.insert(dim, last_dim)

    return tensor.permute(dim_permutation)


def sample_dim(tensor, n, dim, dim_size=None):
    if dim_size is None:
        dim_size = tensor.size(dim)

    index = torch.randperm(dim_size).to(tensor.device)
    index = index[:n]
    return tensor.index_select(dim=dim, index=index)


def sample_padded_sequences(sequences, lens, sample_size):
    seq_dim = 1

    new_shape = list(sequences.shape)
    new_shape[seq_dim] = sample_size

    data = torch.zeros(new_shape).to(sequences.device)
    new_lens = torch.zeros_like(lens).to(lens.device)

    # TODO: Make this vector operation if possible!
    for i, (length, seq) in enumerate(zip(lens, sequences)):
        new_lens[i] = min(length, sample_size)
        sample = sample_dim(
            seq,
            n=new_lens[i],
            dim=0,
            dim_size=length,
        )
        data[i, :sample.size(0)] = sample

    return data, new_lens


def mask_seq_from_lens(tensor, lens):
    # SRC - <https://stackoverflow.com/a/53403392>
    seq_dim = 1
    mask = torch.arange(tensor.size(seq_dim))[None, :] < lens[:, None]
    mask = mask.reshape(*mask.shape, *([1] * (len(tensor.shape) - 2)))
    mask = mask.expand(*tensor.shape)
    mask = mask.to(tensor.device)

    return tensor * mask


@torch.jit.script
def mask_sequence(tensor, mask):
    initial_shape = tensor.shape
    bs, seq = mask.shape
    masked = torch.where(
        mask.reshape(bs * seq, -1),
        tensor.reshape(bs * seq, -1),
        torch.tensor(0, dtype=torch.float32).to(tensor.device),
    )

    return masked.reshape(initial_shape)


def prepare_rnn_state(state, num_rnn_layers):
    """
    RNN cells expect the initial state
    in the shape -> [rnn_num_layers, bs, rnn_state_size]
    In this case rnn_state_size = state // rnn_num_layers.
    The state is distributed among the layers
    state          -> [bs, state_size]
    rnn_num_layers -> int
    """
    return torch.stack(
        state.chunk(torch.tensor(num_rnn_layers), dim=1),
        dim=0,
    )


def time_distribute(module, input=None):
    """
    Distribute execution of module over batched sequential input tensor.
    This is done in the batch dimension to facilitate parallel execution.
    input  -> [bs, seq, *x*]
    module -> something that takes *x*
    return -> [bs, seq, module(x)]
    """
    if input is None:
        return TimeDistributed(module)

    shape = input[0].size() if type(input) is list else input.size()
    bs = shape[0]
    seq_len = shape[1]

    if type(input) is list:
        input = [i.reshape(-1, *i.shape[2:]) for i in input]
    else:
        input = input.reshape(-1, *shape[2:])

    out = module(input)  # should return iterable or tensor

    if hasattr(out, 'view'):
        # assume it is tensor:
        return out.view(bs, seq_len, *out.shape[1:])

    return map(lambda o: o.view(bs, seq_len, *o.shape[1:]), out)


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        shape = input[0].size() if type(input) is list else input.size()
        bs = shape[0]
        seq_len = shape[1]

        if type(input) is list:
            input = [i.reshape(-1, *i.shape[2:]) for i in input]
        else:
            input = input.reshape(-1, *shape[2:])

        out = self.module(input)
        out = out.view(bs, seq_len, *out.shape[1:])

        return out


# @extend(torch.Tensor)
# @property
def to_np(t):
    return t.detach().cpu().numpy()


torch.Tensor.np = property(lambda self: to_np(self))


@extend(torch.Tensor)
def imshow(tensor, padding=3, figsize=(10, 10)):
    """Plot 3-channel image tensorts.
    Types:
        tensor: Tensor[*A, 3, H, W]
        return: AxesImage (matplotlib plot of the batch of images)
    """
    import matplotlib.pyplot as plt

    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)  # [Cols, 1, 3, H, W]

    if tensor.ndim > 3:  # assumes ndim == 5
        nrows = tensor.shape[0]
        tensor = tensor.reshape(-1, *tensor.shape[-3:])
        tensor = torchvision.utils.make_grid(
            tensor, nrow=nrows, padding=padding, normalize=True)

    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.np

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    return ax.imshow(tensor)


@extend(torch.Tensor)
def heatmap(tensor, padding=3, cmap='viridis', figsize=(10, 1)):
    """Plot 1-channel image tensorts as heatmap.
    Types:
        tensor: Tensor[*A, H, W]
        return: AxesImage (matplotlib plot of the batch of images)
    """
    import matplotlib.pyplot as plt

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)  # [Cols, 1, H, W]

    if tensor.ndim > 2:  # assumes ndim == 4
        nrows = tensor.shape[0]
        tensor = tensor.unsqueeze(2)  # [Cols, Rows, 1, H, W]
        # [Cols * Rows, 1, H, W]
        tensor = tensor.reshape(-1, *tensor.shape[-3:])
        # [1, H * Cols, W * Cols]
        tensor = torchvision.utils.make_grid(
            tensor, nrow=nrows, padding=padding, normalize=True)
        tensor = tensor[0]  # [H, W]

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    return ax.imshow(tensor.np, cmap=cmap)


if __name__ == '__main__':
    # Sanity check mask_sequence
    tensor = torch.rand(2, 3, 4)
    mask = torch.rand(2, 3) > 0.5
    masked = mask_sequence(tensor, mask)
    print(mask)
    print(masked)
