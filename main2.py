import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import csv

import numpy as np
import os.path
from os import path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import threading

scaler = torch.cuda.amp.GradScaler()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-f', '--finetune', dest = 'finetune', action = 'store_true',
                    help = "finetune only last layer")
parser.add_argument('-i', '--images', dest = 'write_images', action = 'store_true',
                    help = "write images for validation dataset")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--activations', dest = 'activations', action = 'store_true',
                    help = "don't train, just write activations files")
parser.add_argument('--ood', dest = 'ood', action = 'store_true',
                    help = "don't train, just write logit files")
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-V', '--validation-size', default=0, type=int,
                    metavar='validation_size',
                    help='size of subset of train data set aside for extra validation')

parser.add_argument('--existing-val-split', default=None, type=str, metavar='existing_val_split',
                    help='path to a folder with files val_idx.npy and train_idx.npy specifying '
                                             'training/validation split of training set')

parser.add_argument('--output-folder', default='.', type=str, metavar='output_folder',
                    help='path to a folder in which training outputs will be stored')


best_acc1 = 0


def main():
    # breakpoint()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    valid_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    train_dataset_train_transf = datasets.ImageFolder(
                traindir,
                train_transform
            )

    # needed for validation subset
    train_dataset_valid_transf = datasets.ImageFolder(
                traindir,
                valid_transform
            )


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    classes = train_dataset.classes

    if (not args.activations) and (not args.resume):
        print(f"Setting size of fc layer to {len(classes)} classes")
        old_in = model.fc.in_features
        model.fc = nn.Linear(old_in, len(classes))

    if args.ood:
        assert args.resume
        old_in = model.fc.in_features
        path = os.path.dirname(args.resume)
        # old = dict()
        with open(f"{path}/classes.csv") as f:
            model.fc = nn.Linear(old_in, len(f.readlines()) - 1)

        print("Adjusted the top layer of the network to originally trained class cardinality")
            


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # print(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        # print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if args.finetune:
                args.start_epoch = 0
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    if args.validation_size > 0:
        # Split original train set into a new validation set and remaining train set
        if args.existing_val_split is None:
            # train_targets = train_dataset_train_transf.targets
            # full_train_size = len(train_targets)
            # test_portion = args.validation_size / full_train_size
            # train_idx, valid_idx = train_test_split(
                    # np.arange(full_train_size), test_size=test_portion, shuffle=True,
                    # stratify=train_targets)

            train_targets = np.array(train_dataset_train_transf.targets)
            full_train_size = len(train_targets)
            print(train_targets.shape)

            train_i = np.zeros((full_train_size,), dtype = np.int32)
            for i in range(len(classes)):
                e = np.equal(train_targets, i)
                w = e.nonzero()[0]
                s = np.random.choice(w, size = args.validation_size, replace = False)
                # print(s.shape)
                # print(s)
                train_i[s] = 1

            valid_idx = train_i.nonzero()[0]
            train_i = train_i - 1
            train_idx = train_i.nonzero()[0]
        else:
            print("Loading existing validation split")
            train_idx = np.load(os.path.join(args.existing_val_split, 'train_idx.npy'))
            valid_idx = np.load(os.path.join(args.existing_val_split, 'val_idx.npy'))

        train_subset = torch.utils.data.Subset(train_dataset_train_transf, train_idx)
        valid_subset = torch.utils.data.Subset(train_dataset_valid_transf, valid_idx)

        np.save(os.path.join(args.output_folder, 'train_idx.npy'), np.array(train_idx))
        np.save(os.path.join(args.output_folder, 'val_idx.npy'), np.array(valid_idx))

        val2_loader = torch.utils.data.DataLoader(
               valid_subset, batch_size = args.batch_size, shuffle = False,
               num_workers = args.workers, pin_memory = True, sampler = None)

        # val3_loader = torch.utils.data.DataLoader(
               # torch.utils.data.Subset(train_dataset_train_transf, valid_idx), 
               # batch_size = args.batch_size, shuffle = False,
               # num_workers = args.workers, pin_memory = True, sampler = None)

        train_loader = torch.utils.data.DataLoader(
               train_subset, batch_size=args.batch_size, shuffle=(train_sampler is None),
               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # train2_loader = torch.utils.data.DataLoader(
                # torch.utils.data.Subset(train_dataset_valid_transf, train_idx),
                # batch_size = args.batch_size,
                # shuffle = (train_sampler is None),
                # num_workers = args.workers, pin_memory = True, sampler = train_sampler)

    else:
        print("Using Loader without extra validation split")
        train_loader = torch.utils.data.DataLoader(
              train_dataset_train_transf, batch_size=args.batch_size, shuffle=(train_sampler is None),
              num_workers=args.workers, pin_memory=True, sampler=train_sampler)


#    train_loader = torch.utils.data.DataLoader(
#        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    c2 = zip(classes, val_loader.dataset.classes)
    c3 = map(lambda x: x[1] == x[0], c2)
    assert all(c3)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    @torch.no_grad()
    def write_activations(outdir, loaders, name = "test", layerName = 'avgpool', perClass = 10000):
            print(f"Writing {name} files")

            model.training = False

            activation = []
            lock = threading.Lock()
            def get_activation(name):
                def hook(model, input, output):
                    with lock:
                        activation.append({name: output.detach().clone()})
                return hook
        

            model.module.avgpool.register_forward_hook(get_activation(layerName))
            # model.module.fc.register_forward_hook(get_activation('fc'))
            glob_cts = np.zeros((1000,), dtype = np.int32)
            curr_cts = np.zeros((1000,), dtype = np.int32)

            class Activations:
                def __init__(self):
                    self.lists = dict ( 
                            flatten = [],
                            logits = [],
                            targets = []
                        )
                    cols = []
                    if args.activations:
                        path = os.path.dirname(args.resume)
                        old = dict()
                        with open(f"{path}/classes.csv") as f:
                            rdr = csv.reader(f, delimiter = ',')
                            next(rdr, None) # skip the header
                            for row in rdr:
                                # store class number (out of 1000) as value for each class name (as key)
                                old[row[0]] = row[1]

                        for i, c in enumerate(classes):
                            cols.append(old[c])
                    elif args.ood:
                        path = os.path.dirname(args.resume)
                        old = dict()
                        with open(f"{path}/classes.csv") as f:
                            rdr = csv.reader(f, delimiter = ',')
                            next(rdr, None) # skip the header
                            for row in rdr:
                                old[row[0]] = row[1]

                        for i, key in enumerate(old):
                            cols.append(i)
                    else:
                        for i, _ in enumerate(classes):
                            cols.append(i)

                    self.cols = [int(col) for col in cols]
                    print(self.cols)
                    self.cols_tensor = torch.tensor(self.cols)

                def add(self, key, value):
                    self.lists[key].append(value)

                def add_gpu(self, key, value):
                    if key != 'logits':
                        self.lists[key].append(value.detach().cpu().detach().clone().numpy())
                    else:
                        self.lists[key].append(torch.index_select(value.detach().cpu().detach().clone(), 1, self.cols_tensor).numpy())

                def to_arr(self):
                    if not args.ood:
                        self.flatten_arr = np.concatenate(self.lists['flatten'])
                    self.logits_arr = np.concatenate(self.lists['logits'])
                    self.targets_arr = np.concatenate(self.lists['targets'])
                    # print("flatten: {}\nlogits: {}\ntargets: {}\n".format(self.flatten_arr.shape, self.logits_arr.shape, self.targets_arr.shape))

                def save(self):
                    if not args.ood:
                        np.save(f"{outdir}/{name}_flatten.npy", self.flatten_arr)

                    np.save(f"{outdir}/{name}_logits.npy", self.logits_arr)
                    np.savetxt(f"{outdir}/{name}_targets.csv", self.targets_arr, 
                       fmt = "%d", header = 'target', comments = "")

                    print(self.targets_arr.size)
                    acc = 100. * np.sum(self.targets_arr == np.argmax(self.logits_arr, axis = 1)) /  \
                        self.targets_arr.size
                    print(f"{name} accuracy {acc}%")
                    with open("activ_summary.txt", "a") as fd:
                        fd.write(f"{name},acc1,{acc}\n")
                    # breakpoint()

                def eval_images(self, loader, desc = "    "):
                    model.eval()
                    for (images, target) in tqdm(loader, desc = desc):
                        new_target = target.clone()
                        # print(new_target)
                        for i, k in enumerate(target):
                            curr_cts[k] += 1
                        self.add('targets', new_target)

                        if args.gpu is not None:
                            images = images.cuda(args.gpu, non_blocking=True)
                        if torch.cuda.is_available():
                            target = target.cuda(args.gpu, non_blocking=True)
                        # torch.cuda.empty_cache()
                        with torch.cuda.amp.autocast():
                            pred = model(images)
                        if not args.ood:
                            # print(f"act shape {activation[layerName].shape}")
                            act = torch.nn.parallel.scatter_gather.gather(activation, target_device="cpu")[layerName]
                            activation.clear()
                            # print(f"act shape: {act.shape}")
                            s = act.shape
                            self.add_gpu('flatten', torch.reshape(act, (-1, s[1])))
                        self.add_gpu('logits', pred)
                        del(pred)

                def save_after_extra_rounds(self):
                    N = np.max(self.targets_arr) + 1
                    # print(glob_cts[:N])
                    # print(curr_cts[:N])
                    # print(np.max((perClass * np.ones((N,), dtype = np.float32) - glob_cts[:N]) / curr_cts[:N]))
                    # print((perClass * np.ones((N,), dtype = np.float32) - glob_cts[:N]) / curr_cts[:N])
                    print(self.cols_tensor)
                    todo = int(np.ceil(np.max((perClass * np.ones((N,), dtype = np.float32) - glob_cts[:N]) / curr_cts[:N])))
                    if todo > 0:   
                        print(f"Doing additional {todo} loader generations")
                        for j in range(todo):
                            self.eval_images(loader, desc = f"Extra {j}/{todo}")

                        self.to_arr()
                        acc = 100. * np.sum(self.targets_arr == np.argmax(self.logits_arr, axis = 1)) /  \
                            self.targets_arr.size
                        print(f"Before extra rounds accuracy {acc}")

                    if not args.ood:
                        flatten_fin = np.zeros((N * perClass, np.shape(self.flatten_arr)[1]))
                    logits_fin = np.zeros((N * perClass, np.shape(self.logits_arr)[1]))
                    targets_fin = np.zeros((N * perClass,))

                    for i in range(N):
                        idx = np.asarray(self.targets_arr == i).nonzero()
                        id2 = np.random.choice(idx[0], perClass, replace = False)
                        assert(id2.size == perClass)
                        id3 = np.arange(start = perClass * i, stop = perClass * (i + 1) )
                        assert(id3.size == perClass)
                        if not args.ood:
                            flatten_fin[id3,:] = self.flatten_arr[id2, :]
                        logits_fin[id3,:] = self.logits_arr[id2, :]
                        targets_fin[id3] = self.targets_arr[id2]

                    if not args.ood:
                        self.flatten_arr = flatten_fin
                    self.logits_arr = logits_fin
                    self.targets_arr = targets_fin

                    self.save()
                    

            acts = Activations()

            for (loader, pepochs) in loaders:
                for j in range(pepochs):
                    curr_cts[:] = 0
                    acts.eval_images(loader, desc = f"Iter {j}/{pepochs}")
                    glob_cts += curr_cts
                    # print(torch.cuda.memory_allocated())                                                                   

            acts.to_arr()

            if perClass is None:
                acts.save()
            else:
                acts.save_after_extra_rounds()

            del acts



    def save_predictions():
        # d = f"epoch{epoch}"
        d = "."

        with open("val_files.txt", "w") as f:
            for i, (ff, category) in enumerate(val_loader.dataset.samples, 0):
                f.write(f"{i}, {ff}, {category}\n")

        if args.validation_size > 0:
            write_activations(d, [(val2_loader,1)], name = "valid_from_train", perClass = None)
            #assert(False) # untested
            #write_activations(d, [(train_loader, 9), 
                                  #(train2_loader, 1), 
                                  #(val2_loader, 1), 
                                  #(val3_loader,9)], name = "valid")
        if (len(classes) <= 50):
            write_activations(d, [(train_loader, 11)], name = "valid")
        write_activations(d, [(val_loader, 1)], name = "test", perClass = None)
            

        


    # Before training write classes that we will train on

    f = open("classes.csv", "w")
    f.write("class,id\n")
    for i, cl in enumerate(classes):
        f.write(f"{cl},{i}\n")
    f.close()

    if args.write_images or len(classes) <= 10:
        val_images = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, 
            pin_memory=True
            )

        imglist = []
        for (images, targets) in val_images:
            imglist.append(images.detach().cpu().detach().clone().numpy())

        img_arr = np.concatenate(imglist) 
        print(img_arr.shape)
        np.save("images.npy", img_arr)
        del(val_images)

    if args.activations:
        print("Not going to train, just writing activation files")
        save_predictions()
        return

    if args.ood:
        print("Not going to train, just writing logit files for OOD problem")
        save_predictions()
        return

    if args.finetune:
        lm = [module for module in model.modules()]
        # print(lm[-1])
        del(lm[-1])
        for m in lm:
            # print(m)
            for param in m.parameters():
                # param.requires_grad = False
                pass

        old_in = model.module.fc.in_features
        model.module.fc = nn.Linear(old_in, len(classes)).cuda()


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if (args.validation_size > 0):
            acc1 = validate(val2_loader, model, criterion, epoch, args)
        else:
            acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            # save_predictions(epoch + 1)


    # print(type(model))
    # print(model)

    save_predictions()




def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        assert(output.is_cuda == True)
        assert(output.requires_grad == True)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # loss.backward()
        scaler.scale(loss).backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    with open("valid_summary.txt", "a") as myfile:
        myfile.write("{epoch},{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f},    ".
               format(epoch = epoch, losses = losses, top1 = top1, top5 = top5))


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if args.validation_size == 0:
            print('On test dataset * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        else:
            print('On validation dataset * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    if epoch is not None:
        with open("valid_summary.txt", "a") as myfile:
            myfile.write("{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f}\n".
                    format(losses = losses, top1 = top1, top5 = top5))

    return top1.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    # directory = f"epoch{state['epoch']}"
    # try:
        # os.mkdir(directory)
    # except OSError as error:
        # print(error)
    # torch.save(state, f"{directory}/{filename}")
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(f"{directory}/{filename}", 'model_best.pth.tar')
        if args.validation_size > 0:
            shutil.copyfile(filename, 'model_best_valid.pth.tar')
        else:
            shutil.copyfile(filename, 'model_best_test.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].sum(dtype = float).expand(1)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
