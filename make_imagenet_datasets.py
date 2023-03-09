import os


def train(d):
    assert os.path.isdir(d)
    arch = "resnet18"

    os.chdir(d)

    N = 1 if d == "imagenet-50" else 20

    for i in range(N):
        sdir = f"net{i+1}"
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
            os.chdir(sdir)
            print(f"python  ../../main2.py --resume ../../networks/{sdir}/checkpoint.pth.tar --activations -j 6 -p 50 -a {arch} ..")
            os.system(f"python  ../../main2.py --resume ../../networks/{sdir}/checkpoint.pth.tar --activations -j 6 -p 50 -a {arch} ..")
            os.chdir("..")

    if d == "imagenette":
        for i in range(N):
            sdir = f"eva{i+1}"
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
                os.chdir(sdir)
                arch = "resnet18"
                print(f"python  ../../main2.py -V 200 -j 10 -p 50 -a {arch} ..")
                os.system(f"python  ../../main2.py -V 200 -j 10 -p 50 -a {arch} ..")
                os.chdir("..")

    os.chdir("..")


def main20():
    for d in ["imagewoof", "imagenette", "imagenet-50"]:
        train(d)

def main50():
    if not os.path.isdir("imagenet-50/eva1"):
        os.mkdir("imagenet-50/eva1")
        os.chdir("imagenet-50/eva1")
        print(f"python3  ../../main2.py -b 128 -V 200 -j 8 -p 50 -a resnet18  ..")
        os.system(f"python3  ../../main2.py -b 128 -V 200 -j 8 -p 50 -a resnet18  ..")
