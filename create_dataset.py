import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

class FFTDataset(Dataset):
    def __init__(self, root, train=False, transform=transform, target_transform=None, download=False):
        self.dataset = CIFAR10(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.compute_fft = self.compute_fft
        self.split_fft = self.split_fft

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_fft = self.compute_fft(img)
        magnitude, phase = self.split_fft(x_fft)
        return (magnitude, phase, img), label

    def compute_fft(self, x):
        x_fft = torch.fft.fftn(x, dim=[-2, -1])
        return x_fft

    def split_fft(self, x_fft):
        x_magnitude = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        return x_magnitude/1000., (x_phase/3.1416 + 1)/2.

