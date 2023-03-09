import os


if not os.path.isdir("cifar10"):
    os.mkdir("cifar10")

os.chdir("cifar10")

for i in range(20):
        if not os.path.isdir(f"net{i+1}"):
            os.mkdir(f"net{i+1}")
        os.system(f"python  ../train_cifar.py --val_size=0 --out_dir net{i+1}")

