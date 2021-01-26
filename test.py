import os
from shutil import copyfile

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
        pred_outputs = []
        pred_outputs = torch.tensor(pred_outputs).to(config.device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            pred_outputs = torch.cat([pred_outputs, pred], dim=0)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    return test_loss, acc, pred_outputs


def dataset(data_dir):
    resize = transforms.Resize((256, 256))
    grayscale = transforms.Grayscale(num_output_channels=1)
    data_transform = transforms.Compose([resize, grayscale, transforms.ToTensor()])

    dataset = ImageFolder(data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return dataloader


def check_predictions(pred, dataset):
    """예측값 별로 test 이미지 split하여 폴더별 저장
    args: test prediction 값, loader의 dataset
    """
    pred_dir = "./dataset/" + "/pred"
    authentic_dir = pred_dir + "/authentic/"
    manipulated_dir = pred_dir + "/manipulated/"
    if not os.path.isdir(authentic_dir):
        os.makedirs(authentic_dir)
    if not os.path.isdir(manipulated_dir):
        os.makedirs(manipulated_dir)
    for i in range(len(pred)):
        base_name = dataset.imgs[i][0].split("/")[4]
        if pred[i].item() == 0:
            copyfile(dataset.imgs[i][0], authentic_dir + base_name)
        else:
            copyfile(dataset.imgs[i][0], manipulated_dir + base_name)

if __name__ == "__main__":
    # 최종 모델 로드
    # model = torch.load(config.model_path + "MISLnet.pt", map_location=config.device)
    # model.eval()

    # 체크포인트 모델 로드
    model = MISLnet().to(config.device)
    if torch.cuda.is_available():
        checkpoint = torch.load(config.model_path + "60_model.pt")
    else:
        checkpoint = torch.load(config.model_path + "60_model.pt", map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = dataset(config.test_dataset)
    test_loss, acc, pred = test(model, loader)
    # pred에 따라 파일 분류
    check_predictions(pred, loader.dataset)
    print("-> testing loss={} acc={}".format(test_loss, acc))
