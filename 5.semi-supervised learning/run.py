import torch, time

from Utils.utils import *
from dataloaders import *
from Task.trainer import *
from model import ResNet, ResidualBlock

import warnings
warnings.filterwarnings(action='ignore')

def main(args):
    
    # set seed & device
    set_seed(args.seed)
    device = args.device
    total_l = args.num_l
    
    # load CIFAR10
    trainset = torch_datasets.CIFAR10(root='./Data', train=True, download=False)
    testset = torch_datasets.CIFAR10(root='./Data', train=False, download=False)
    
    X_train, y_train = trainset.data, trainset.targets
    X_test, y_test = testset.data, testset.targets
    
    # initialize dataloaders & pseudo labels
    loaders, indices = init_dataloaders(args, X_train, y_train, X_test, y_test)
    y_pseudo = []
    
    # start self-training
    print(f'>>> Self-Training Start...')
    iter_acc = []
    num_label_list = [total_l]
    for i in range(1, args.iteration+1):
        
        # define model, optimizer, criterion
        model = ResNet(ResidualBlock, [2,2,2]).to(device)
        opt = torch.optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        ## Train Model
        print(f'>>> {i}th Iteration...')
        for epoch in range(1, args.epochs+1):
            since = time.time()
            train_loss, train_acc = train(args, model, opt, criterion, loaders['labeled'])
            train_time = round(time.time()-since,2)

            print(f'Iteration : {i} | Epoch : {epoch} | Num Labels : {total_l}')
            print(f'train loss : {round(train_loss,4)} | train acc : {train_acc}% | train time : {train_time}sec\n')
            

        pseudo_indices, pseudo_labels = get_pseudo_label(args, model, loaders['unlabeled'])
        num_pseudo, total_l, total_u = update_dataloaders(args, X_train, y_train, y_pseudo, 
                                                          loaders, indices, pseudo_indices, pseudo_labels)

        labeled_rate = round(total_l/(total_l+total_u)*100,2)
        num_label_list.append(total_l)
        print('Pseudo labeling & update DataLoaders')
        print(f'# of new pseudo labels : {num_pseudo} | labeled : unlabeled = {labeled_rate}% : {round(100-labeled_rate,2)}%\n')
    
        print(f'>>> {i}th Iteration Test Start...')
        test_acc = test(args, model, loaders['test'])
        iter_acc.append(test_acc)
        print(f'Test Accuracy : {test_acc}')
    
    acc = round(np.mean(iter_acc),2)
    
    return acc, iter_acc, num_label_list

if __name__ == '__main__':
    from Utils.config import Config
    
    args = Config()
    main(args)
        
        
            
        
            
            
    
    
    
    
    