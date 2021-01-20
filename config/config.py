import os
import torch


class config(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_class = 2
    total_epoch = 60
    batch_size = 64
    learning_rate = 0.001

    model_path = "./model/checkpoint/"
    train_dataset = "./dataset/train"
    test_dataset = "./dataset/test"
