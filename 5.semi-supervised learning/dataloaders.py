import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as torch_datasets

from datasets import CIFAR10_Dataset

def init_dataloaders(args, X_train, y_train, X_test, y_test):
    
    # define label & unlabel data
    total_indices = list(range(args.train_imgs))
    label_indices = random.sample(total_indices, args.num_l)
    unlabel_indices = list(set(total_indices)-set(label_indices))
    
    X_labeled, y_labeled = X_train[label_indices], np.array(y_train)[label_indices].tolist()
    X_unlabeled, y_unlabeled = X_train[unlabel_indices], np.array(y_train)[unlabel_indices].tolist()
    
    print(f'# of train data : {args.train_imgs} | # of test data : {args.test_imgs}')
    print(f'# of labeled data in trainset : {len(label_indices)} | # of unlabeled data in trainset : {len(unlabel_indices)}')
    print(f'labeled : unlabeled = {round(len(label_indices)/500,2)}% : {100-round(len(label_indices)/500,2)}%\n')
    
    # make datasets
    labeled_set = CIFAR10_Dataset(X_labeled, y_labeled)
    unlabeled_set = CIFAR10_Dataset(X_unlabeled, y_unlabeled)
    test_set = CIFAR10_Dataset(X_test, y_test)
    
    # make dataloader
    label_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabel_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    loaders = dict(labeled=label_loader,
                   unlabeled=unlabel_loader,
                   test=test_loader)
    
    indices = dict(total=total_indices,
                   labeled=label_indices,
                   unlabeled=unlabel_indices,
                   pseudo=[])
    
    return loaders, indices

def update_dataloaders(args, X_train, y_train, y_pseudo, 
                       loaders, indices, pseudo_indices, pseudo_labels):
    
    # update pseudo labeled & unlabeled indices
    indices['pseudo'] += pseudo_indices
    indices['unlabeled'] = list(set(indices['unlabeled'])-set(pseudo_indices))
    
    # define labeled & pseudo labeled & unlabeled data
    X_labeled, y_labeled = X_train[indices['labeled']], np.array(y_train)[indices['labeled']].tolist()
    X_unlabeled, y_unlabeled = X_train[indices['unlabeled']], np.array(y_train)[indices['unlabeled']].tolist()
    X_pseudo = X_train[indices['pseudo']]
    y_pseudo += pseudo_labels # update pseudo labels
    
    # make datasets
    labeled_set = CIFAR10_Dataset(X_labeled, y_labeled)
    unlabeled_set = CIFAR10_Dataset(X_unlabeled, y_unlabeled)
    pseudo_set = CIFAR10_Dataset(X_pseudo, y_pseudo)
    
    # concat labeled & pseudo labeled set
    if len(indices['pseudo']) != 0:
        new_labeled_set = ConcatDataset([labeled_set, pseudo_set])
    else:
        new_labeled_set = labeled_set

    # make new dataloader
    loaders['labeled'] = DataLoader(new_labeled_set, batch_size=args.batch_size, shuffle=True)
    loaders['unlabeled'] = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    
    # number of labeled & unlabeled data
    num_pseudo = len(pseudo_indices)
    total_l = len(indices['labeled'])
    total_u = len(indices['unlabeled'])
    
    return num_pseudo, total_l, total_u
    