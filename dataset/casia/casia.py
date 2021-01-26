import os
from PIL import Image
from shutil import copyfile

def to_png():
    path = "."
    real_path ='./Au'
    probe_path ='./Tp'
    if not os.path.exists(real_path):
        os.mkdir(real_path)
    if not os.path.exists(probe_path):
        os.mkdir(probe_path)        

    for root, dirs, files in os.walk(path, topdown=False):
        dst = ''
        for name in files:
            filepath = os.path.join(root, name)
            filename = os.path.basename(filepath)
            ext = os.path.splitext(os.path.join(root, name))[1].lower()
            if filename.find('Au') > -1:
                dst = './authentic'
            elif filename.find('Tp') > -1:
                dst = './manipulated' 
            else:
                continue 
            try:
                if ext == '.tiff' or ext == '.tif' or ext  =='.jpg' or ext =='.jpeg':
                    outfile = os.path.join(dst,os.path.splitext(name)[0]+'.png')
                    im = Image.open(filepath)
                    im.thumbnail(im.size)
                    im.save(outfile,'png',quality=100)
            except:
                pass

def split_dataset():
    # authentic
    authentic_path = './authentic/'
    for root, dirs, files in os.walk(authentic_path):
        train_files = files[:4500]
        test_files = files[4500:5000]
    for file in train_files:
        copyfile(authentic_path + file, '../train/authentic/' + file)
    for file in test_files:
        copyfile(authentic_path + file, '../test/authentic/' + file)

    # manipulated
    manipulated_path = './manipulated/'
    for root, dirs, files in os.walk(manipulated_path):
        train_files = files[:4500]
        test_files = files[4500:5000]
    for file in train_files:
        copyfile(manipulated_path + file, '../train/manipulated/' + file)
    for file in test_files:
        copyfile(manipulated_path + file, '../test/manipulated/' + file)

if __name__ == "__main__":
    to_png()
    split_dataset()