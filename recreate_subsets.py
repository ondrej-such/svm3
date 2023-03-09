import os
import os.path
import csv
import sys

def create_dataset(cd, f):
    [dname,_] = os.path.splitext(f)
    print(f"{f}, {dname}")

    os.mkdir(f"{dname}")
    os.mkdir(f"{dname}/train")
    os.mkdir(f"{dname}/val")

    with open(f"imagenet_subsets/{f}") as cf:
        rdr = csv.reader(cf)
        hdr = rdr.__next__()
        for row in rdr:
            [D, _] = row
            src1 = f"{cd}/train/{D}"
            print(src1)
            dst1 = f"{dname}/train/{D}"
            os.symlink(src1, dst1)
            os.symlink(f"{cd}/val/{D}", f"{dname}/val/{D}")



def main():
    files = os.listdir("imagenet_subsets")
    print(files)
    for f in files:
        create_dataset(sys.argv[1], f"{f}")

main()
