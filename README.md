# Introduction

This repository contains code necessary to build and evaluate intermediate SVM models (both incremental and bipartite). It allows to reproduce tables and graphs from [Bridging performance gap between minimal and maximal SVM models], provides crucial datasets and extended analysis results. Not all intermediate files are included, but instructions below should guide you how to rerun experiments and analyses.

# Software preliminaries

You will most likely need Linux machine to run the code. Windows subsystem WSL2 should work as well, but on Windows itself you may encounter compilation errors.

To run analyses you will need working R and Python installation. 

## R setup
From R you need the following packages:

<code>
Rcpp
ape
caret
dplyr
e1071
ggplot2 
grid
parallel
readr
reticulate
tidyr
tidyverse
</code>

In order to correctly compile Rcpp code you should add the following to your  ~/.R/Makevars

```
PKG_CXXFLAGS = -Wno-ignored-attributes
PKG_LIBS = -lquadmath -llapack
LDLIBS = -llapack
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/lapack
```

Alternatively you can copy Makevars file included at the top level of the repository.

## Python setup

We used Python 3.8.5 with standard torch, torchvision libraries. Detailed package list is in the file requirements.txt.


# Rerunning analyses and experiments

There are three levels at which you can reproduce our results.

## Run analyses from included datasets

The markdown file analyses.Rmd can be used as a starting point for analyses of included datasets that are in data.frames
subdirectory.  You can open it in RStudio and build analyses.html report. 


## Train neural networks from scratch

You will need to create symbolic link named imagenet pointing to folder with Imagenet dataset. Then simply issue

```
make clean; make all
```

Note that it may a long time (possibly several weeks). Multiple GPU will speed up training.
