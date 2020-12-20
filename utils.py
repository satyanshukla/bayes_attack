# pylint: disable=E1101, E0401, E1102
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


def load_mnist_data():
    dataset = dsets.MNIST(root='./data/mnist', train=False,
                          transform=transforms.ToTensor())
    return dataset


def load_cifar10_data():
    dataset = dsets.CIFAR10('./data/cifar10-py', download=True,
                            train=False, transform=transforms.ToTensor())
    return dataset


def load_imagenet_data(size1=256, size2=224):
    dataset = dsets.ImageFolder(
        '../../../../../elm/ILSVRC2012/val/',
        transform=transforms.Compose([
            transforms.Resize(size1),
            transforms.CenterCrop(size2),
            transforms.ToTensor(),
        ]))

    return dataset


def proj(pert, eps, inf_norm, discrete):
    # project image into epsilon ball (either L_inf norm or L_2 norm)
    # if discrete=True, project into the boundary of the ball instead of the ball
    if inf_norm:
        if discrete:
            return eps * pert.sign()
        else:
            return pert.clamp(-eps, eps)
    else:
        pert_norm = torch.norm(pert.view(pert.shape[0], -1), dim=1)
        pert_norm = torch.where(pert_norm > eps, pert_norm / eps,
                                torch.ones_like(pert_norm))
        return pert / pert_norm.view(-1, 1, 1, 1)


def latent_proj(pert, eps):
    # project the latent variables (i.e., FFT variables)
    # into the epsilon L_2 ball
    pert_norm = torch.norm(pert, dim=1) / eps
    return pert.div_(pert_norm.view(-1, 1))


def fft_transform(pert):
    # single channel fft transform
    res = torch.zeros(pert.shape[0], 1, 28, 28)
    t_dim = int(np.sqrt(pert.shape[1]))
    for i in range(pert.shape[0]):
        t = torch.zeros((28, 28, 2))
        t[:t_dim, :t_dim] = pert[i].view(t_dim, t_dim, 2)
        res[i, 0] = torch.irfft(t, 2, normalized=True, onesided=False)
    return res


def fft_transform_mc(pert, dataset_size, channel, cosine, sine):
    # multi-channel FFT transform (each channel done separately)
    # performs a low frequency perturbation (set of allowable frequencies determined by shape of pert)
    res = torch.zeros(pert.shape[0], channel, dataset_size, dataset_size)
    for i in range(pert.shape[0]):
        t = torch.zeros((channel, dataset_size, dataset_size, 2))
        if cosine and sine:
            t_dim = int(np.sqrt(pert.shape[1] / channel / 2))
            t[:, :t_dim, :t_dim, :] = pert[i].view(channel, t_dim, t_dim, 2)
        elif cosine:
            t_dim = int(np.sqrt(pert.shape[1] / channel))
            t[:, :t_dim, :t_dim, 0] = pert[i].view(channel, t_dim, t_dim)
        elif sine:
            t_dim = int(np.sqrt(pert.shape[1] / channel))
            t[:, :t_dim, :t_dim, 1] = pert[i].view(channel, t_dim, t_dim)
        res[i] = torch.irfft(t, 3, normalized=True, onesided=False)
    return res


def transform(pert, dset, arch, cosine, sine):
    if dset == 'cifar10':
        return fft_transform_mc(pert, 32, 3, cosine, sine)
    elif dset == 'imagenet':
        if arch == 'inception_v3':
            return fft_transform_mc(pert, 299, 3, cosine, sine)
        else:
            return fft_transform_mc(pert, 224, 3, cosine, sine)
    elif dset == 'mnist':
        return fft_transform(pert)
