class Config(object):
    def __init__(self) -> None:
        
        # experiment hyperparameters
        self.seed = 2022                         # seed
        self.device = 'cuda:0'
        
        # data hyperparameters
        self.train_imgs = 50000                  # number of train imgs
        self.test_imgs = 10000                   # number of test imgs
        self.num_l = 25000                       # number of labeled data
        self.num_u = self.train_imgs - self.num_l# number of unlabeled data
        
        # train hyperparameters
        self.batch_size = 128                    # batch size
        self.lr = 3e-4                           # learning rate
        self.weight_decay = 5e-4                 # weight decay
        self.epochs = 50                         # train epoch
        
        # self train hyperparameters
        self.iteration = 3                       # self training iteration
        
        # pseudo label strategy hyperparameters
        self.strategy = 'threshold'              # pseudo labeling strategy ['threshold', 'top_k', 'both']
        self.threshold = 0.95                    # threshold for pseudo labeling
        self.top_k = 2500                        # top-k for pseudo labeling

        
    