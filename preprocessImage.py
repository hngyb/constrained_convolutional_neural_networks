import cv2
import os
import numpy as np
from PIL import Image
from shutil import copyfile
from scipy import misc

def Crop_and_Grayscale(data_dir):

    if not os.path.isdir("./crop_and_grayscale"):
        os.makedirs("./crop_and_grayscale")

    n = 1280
    m = 1280

    for root, dirs, files in os.walk(data_dir, topdown=False):
        k = 0
        for name in files:
            try:
                filepath = os.path.join(root, name)
                img = cv2.imread(filepath)
                i1_start = (img.shape[0] - n) // 2
                i1_stop = i1_start + n
                i2_start = (img.shape[1] - m) // 2
                i2_stop = i2_start + m
                img = img[i1_start:i1_stop, i2_start:i2_stop, :]
                img_gr = img[:, :, 1]  # extract green channel
                if img_gr.shape == (1280,1280):
                    for i in range(5):
                        for j in range(5):
                            img_gr_crop = img_gr[256 * j : 256 * (j + 1), 256 * i : 256 * (i + 1)]
                            cv2.imwrite("./NIST16_crop_and_grayscale/{}_{}_{}.jpg".format(k, i, j), img_gr_crop)
            except:
                pass
            k += 1

def split_dataset(dir_path):
    for root, dirs, files in os.walk(dir_path):
        train_files_au = files[:6000]
        train_files_tp = files[6000:12000]
        test_files_au = files[12000:14950]
        test_files_tp = files[14950:]
    for file in train_files_au:
        copyfile(dir_path + file, './train/authentic/' + file)
    for file in train_files_tp:
        copyfile(dir_path + file, './train/manipulated/' + file)
    for file in test_files_au:
        copyfile(dir_path + file, './test/authentic/' + file)
    for file in test_files_tp:
        copyfile(dir_path + file, './test/manipulated/' + file)
    
def GaussianBlur(data_dir):
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            img = cv2.imread(filepath)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(data_dir + name, img)

def MedianBlur(data_dir):
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            img = cv2.imread(filepath)
            img = cv2.medianBlur(img,5)
            cv2.imwrite(data_dir + name, img)

def JPEG(data_dir):
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            img = cv2.imread(filepath)
            cv2.imwrite(data_dir + name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

def Resampling(data_dir):
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            img = cv2.imread(filepath)
            img = misc.imresize(img, 1.5)
            cv2.imwrite(data_dir + name, img)


if __name__ == "__main__":
    # Crop_and_Grayscale("./NIST16")
    split_dataset('./NIST16_crop_and_grayscale/')
    # GaussianBlur("./train/manipulated/")
    # GaussianBlur("./test/manipulated/")
    # MedianBlur("./train/manipulated/")
    # MedianBlur("./test/manipulated/")
    # JPEG("./train/manipulated/")
    # JPEG("./test/manipulated/")
    Resampling("./train/manipulated/")
    Resampling("./test/manipulated/")