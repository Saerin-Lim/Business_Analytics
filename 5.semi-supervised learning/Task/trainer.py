import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

def train(args, model, opt, criterion, train_loader):
    model.train()
    softmax = nn.Softmax()
    
    train_loss = 0.
    
    y_true, y_pred = [], []
    
    # supervised learning with labeled data + pseudo labeled data
    for idx, batch in enumerate(train_loader):
        inputs = batch['X'].to(args.device) # B, 3, 32, 32
        targets = batch['y'].to(args.device) # B,
        
        # get pred
        preds = model(inputs) # B, 10
        
        # get cross entropy loss
        loss = criterion(preds, targets)
        
        # backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
        
        y_true += targets.detach().cpu().tolist()
        y_pred += softmax(preds).argmax(dim=-1).detach().cpu().tolist()
        
    train_loss /= (idx+1)
    train_acc = round(accuracy_score(y_true, y_pred)*100, 2)
    
    return train_loss, train_acc

def test(args, model, test_loader):
    model.eval()
    softmax = nn.Softmax()
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['X'].to(args.device) # B, 3, 32, 32
            
            # get pred
            probs = softmax(model(inputs)) # B, 10
            y_pred += probs.argmax(dim=-1).detach().cpu().tolist() # B,
            
            y_true +=  batch['y'].detach().tolist() # B,
        
    acc = round(accuracy_score(y_true, y_pred)*100, 2)
    
    return acc

def get_pseudo_label(args, model, unlabeled_loaders):
    model.eval()
    softmax = nn.Softmax()
    
    
    data_indices = [] # total unlabeled data index
    max_probs = [] # max prob list
    pred_labels = [] # pseudo label list
    
    # get probability of unlabeled data
    with torch.no_grad():
        for batch in unlabeled_loaders:
            
            data_indices += batch['idx'].tolist()
            inputs = batch['X'].to(args.device) # B, 3, 32, 32
            
            # get class distribution using softmax function
            preds = softmax(model(inputs)) # B, 10
            
            # get max prob and pseudo label
            probs, pseudo = preds.max(dim=1)
            
            max_probs += probs.detach().cpu().tolist()
            pred_labels += pseudo.tolist()
    
    # get pseudo label
    if args.strategy == 'threshold':
        candidates = np.argwhere(np.array(max_probs) > args.threshold).reshape(-1)
        
    elif args.strategy == 'top_k':
        candidates = np.argsort(max_probs)[::-1][:args.top_k]
        
    else:
        candidates_1 = np.argwhere(np.array(max_probs) > args.threshold).reshape(-1)
        candidates_2 = np.argsort(max_probs)[::-1][:args.top_k]
        
        # data taht satisfy both threshold and top_k
        candidates = np.intersect1d(candidates_1, candidates_2)
    
    pseudo_indices = np.array(data_indices)[candidates].tolist()
    pseudo_labels = np.array(pred_labels)[candidates].tolist()
    
    return pseudo_indices, pseudo_labels
    
    
            
            
    
    
    
    
        
        