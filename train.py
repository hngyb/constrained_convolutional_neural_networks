import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from config import config
from model.MISLnet import MISLnet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


def train(model, loader):
    """MISLnet 학습
    args: model ,loader
    return: 1에폭마다 모델 체크포인트 저장
    """
    # 설정
    writer = SummaryWriter()
    device = config.device
    learning_rate = config.learning_rate
    total_epoch = config.total_epoch
    batch_size = config.batch_size
    model_path = config.model_path

    print("start train model")

    max_step = 0
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=6, gamma=0.5, verbose=True
    )  # 스케줄러: 6에폭마다 lr * 0.5
    for epoch in range(total_epoch):
        acc_list = []
        for step, (x, y) in enumerate(loader):
            max_step = max(max_step, step)
            global_step = epoch * max_step + step
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y).to(device)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct = pred.eq(y.data.view_as(pred)).cpu().sum().item()
            acc = 100.0 * (correct / batch_size)
            acc_list.append(acc)
            print("epoch: {} step: {} done".format(epoch, step))
        print(
            "-> training epoch={:d} loss={:.3f} acc={:.3f}%".format(epoch, loss, np.mean(acc_list))
        )

        # 텐서보드 기록
        # 텐서보드 커맨드: tensorboard --logdir=runs
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", np.mean(acc_list), epoch)
        writer.close()

        # 모델 체크포인트 저장
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            config.model_path + "{}_model.pt".format(epoch),
        )

    # 최종 모델 저장
    torch.save(model, model_path + "MISLnet.pt")
    print("train done")


def dataset(data_dir):
    """dataloader 구성(resize(256*256), grayscale(channel=1))
    class: authentic, manipulated
    args: 데이터셋 폴더(root 폴더 안에 class별 폴더 구성)
    return: dataloader
    """
    resize = transforms.Resize((256, 256))
    grayscale = transforms.Grayscale(num_output_channels=1)
    data_transform = transforms.Compose([resize, grayscale, transforms.ToTensor()])

    dataset = ImageFolder(data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    classes = ["authentic", "manipulated"]

    return dataloader


if __name__ == "__main__":
    model = MISLnet().to(config.device)
    loader = dataset(config.train_dataset)
    train(model, loader)
