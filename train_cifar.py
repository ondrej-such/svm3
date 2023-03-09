from core import *
from torch_backend import *
from dawn_utils import net, tsv
import argparse
import os.path
import sys



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--val_size', type=int, default=5000)
parser.add_argument('--out_dir', type=str, default='')

# from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

     
def main():  
    args = parser.parse_args()
    val_size = args.val_size    
    print(f"Size of validation subset is {val_size}")
    
    print('Downloading datasets')
    dataset = cifar10(args.data_dir)

    epochs = 24
    # for debugging use this!
    # epochs = 1
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = Network(net()).to(device).half()
    loss = x_ent_loss
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(dataset['valid']['targets']) % batch_size]:
        warmup_cudnn(model, loss, random_batch(size))
    
    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)
    
    print('Preprocessing training data')
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]

    # from https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
    # val_size = 5000
    # val_size is set as main argument
    if val_size > 0:
        train_size = len(dataset['train']['targets']) - val_size
        print(f"Aiming to get ({train_size}, {val_size}) train-validation split")
        train_d, val_d, train_t, val_t  = train_test_split(
                dataset['train']['data'], dataset['train']['targets'], 
                train_size = train_size,
                test_size = val_size,
                stratify = dataset['train']['targets']
                )
        valid_set = list(zip(*preprocess({"data" : val_d, "targets": val_t}, 
            [partial(pad, border=4)] + transforms).values()))
    else:
        train_d, train_t = dataset['train']['data'], dataset['train']['targets']
    # train_ds, val_ds = random_split(dataset['train'], [train_size, val_size])
    # dataset['train'] = train_ds
    # dataset['valid'] = val_ds


    # train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    train_set = list(zip(*preprocess({"data" : train_d, "targets": train_t}, 
            [partial(pad, border=4)] + transforms).values()))

    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    
    train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, drop_last=False)

    opts = [SGD(trainable_params(model).values(), {
        'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]
    logs, state = Table(), {MODEL: model, LOSS: loss, OPTS: opts}
    for epoch in range(epochs):
        logs.append(union({'epoch': epoch+1}, train_epoch(state, timer, train_batches, test_batches)))

    with open(os.path.join(os.path.expanduser(args.log_dir), f'logs{val_size}.tsv'), 'w') as f:
        f.write(tsv(logs.log))        

    model.training = False                                                                                     


    def get_prediction(loaders = [(test_batches, 1)], name = "test", outdir = ""):
        print(f"Writing {name} files")
        pred_flatten = []
        pred_logits = []
        targets = []


        for (loader, pepochs) in loaders:
                                                                                                                   
            for j in range(pepochs):
                for i, batch in enumerate(loader):                                                                   
                    # print("---")                                                                                           
                    # print(torch.cuda.memory_allocated())                                                                   
                    torch.cuda.empty_cache()                                                                               
                    pred = model.forward(batch)                                                                            
                    # print(batch.keys())
                    target = batch['target']
                    #for k,v in pred.items():                                                                              
                    #    print(k, type(v), v.shape)                                                                                 
                    pred_flatten.append(pred['flatten'].cpu().detach().numpy())                                          
                    pred_logits.append(pred['logits'].cpu().detach().numpy())                                          
                    targets.append(target.cpu().detach().clone().numpy())                                          
                    del(pred)   

        flatten_arr = np.concatenate(pred_flatten)
        logits_arr = np.concatenate(pred_logits)
        targets_arr = np.concatenate(targets)
        np.save(f"{outdir}/{name}_flatten.npy", flatten_arr)
        np.save(f"{outdir}/{name}_logits.npy", logits_arr)
        np.savetxt(f"{outdir}/{name}_targets.csv", targets_arr, fmt = "%d", header = 'target', comments = "")

        # train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, smet_random_choices=True, drop_last=True)

    trainLoaderExt = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=False)
    trainLoader = DataLoader(train_set, batch_size, shuffle=False, drop_last=False)

    # valid cases - a mix of untransformed and transformed images from the train set
    # valid2 cases - only transformed images from the train set 
    val_list = [(trainLoaderExt, 1),
                (trainLoader, 1)
               ]

    if len(args.out_dir) == 0:
        outdir = f"v{val_size}"
    else:
        outdir = args.out_dir

    if val_size > 0:
        validLoaderExt = DataLoader(Transform(valid_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=False)
        validLoader = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False)
        val_list = val_list + [ (validLoaderExt, 1), 
                                (validLoader, 1)
                              ] 

        get_prediction([(trainLoaderExt, 2), (validLoaderExt,2)], name = "valid2", outdir = outdir)
    else:
        get_prediction([(trainLoaderExt, 2)], name = "valid2", outdir = outdir)


    get_prediction(val_list, name = "valid", outdir = outdir)
    get_prediction([(test_batches, 1)], name = "test", outdir = outdir)

main()

