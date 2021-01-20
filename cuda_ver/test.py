import os

from PIL.Image import NEAREST
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import config
from model.MISLnet import MISLnet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def test(model, loader):
    device = config.device
    print("start test!")
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    return test_loss, acc


def dataset(data_dir):
    resize = transforms.Resize((256, 256))
    grayscale = transforms.Grayscale(num_output_channels=1)
    data_transform = transforms.Compose([resize, grayscale, transforms.ToTensor()])

    dataset = ImageFolder(data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    classes = ["authentic", "manipulated"]
    return dataloader


if __name__ == "__main__":
    # 체크포인트 모델 로드
    # model = MISLnet().to(config.device)
    # checkpoint = torch.load(config.model_path + "50_model.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    # 최종 모델 로드
    model = torch.load(config.model_path + "MISLnet.pt")
    model.eval()

    loader = dataset(config.test_dataset)
    test_loss, acc = test(model, loader)
    print("-> testing loss={} acc={}".format(test_loss, acc))
