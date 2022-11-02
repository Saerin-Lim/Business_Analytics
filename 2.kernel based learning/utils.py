import random
import torch, torchvision
import pandas as pd
import numpy as np
import sklearn.datasets
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

def set_seed(seed:int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def torch_to_numpy(data):
    x = data.data
    x = torch.flatten(x, 1, 2).detach().numpy()
    x = x/255
    
    y = data.targets.detach().numpy()
    
    set_seed(2022)
    
    for i in range(10):
        sampling_idx = random.sample(list(np.argwhere(y==i).reshape(-1)), 150)
        
        if i == 0:
            new_x = x[sampling_idx]
            new_y = y[sampling_idx]
        else:
            new_x = np.concatenate((new_x, x[sampling_idx]))
            new_y = np.concatenate((new_y, y[sampling_idx]))
            
    
    return new_x, new_y
    
def simple_preprocessing(data:pd.DataFrame, target_cols:str):
    data = data.dropna()
    y = data[target_cols]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    x = data.drop([target_cols], axis=1)
    x = pd.get_dummies(x, drop_first=True).to_numpy()
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    return x, y

def load_data(name:str):
    
    if name == 'iris':
        data = sklearn.datasets.load_iris()
        x = data.data
        y = data.target
    
    elif name == 'breast_cancer':
        data = sklearn.datasets.load_breast_cancer()
        x = data.data
        y = data.target
        
    elif name == 'wine':
        data = sklearn.datasets.load_wine()    
        x = data.data
        y = data.target
        
    elif name == 'penguins':
        data = sns.load_dataset(name)
        
        x,y = simple_preprocessing(data, 'species')
    
    elif name == 'titanic':
        data = sns.load_dataset(name)
        
        x,y = simple_preprocessing(data, 'survived')
    
    elif name == 'algerian_forest_fires':
        data = pd.read_csv('./data/algerian_forest_fires.csv', skiprows=1)
        data.drop(['day','month','year'], axis=1, inplace=True)
        data.columns = data.columns.str.strip()
        data['Classes'] = data['Classes'].str.strip()
        x,y = simple_preprocessing(data, 'Classes')
    
    elif name == 'breast_cancer_coimbra':
        data = pd.read_csv('./data/breast_cancer_coimbra.csv')
        x,y = simple_preprocessing(data, 'Classification')
    
    elif name == 'heart_failure_clinical_records':
        data = pd.read_csv('./data/heart_failure_clinical_records.csv')
        x,y = simple_preprocessing(data, 'DEATH_EVENT')
    
    elif name == 'seeds':
        data = pd.read_csv('./data/seeds.txt', sep='\t', header=None)
        x,y = simple_preprocessing(data, 7)
    
    elif name == 'HCV':
        data = pd.read_csv('./data/HCV.csv').drop(['Unnamed: 0'], axis=1)
        x,y = simple_preprocessing(data, 'Category')
        
    elif name == 'breast_tissue':
        data = pd.read_csv('./data/breast_tissue.csv').drop(['Unnamed: 10'], axis=1)
        x,y = simple_preprocessing(data, 'Class')
    
    elif name == 'digits':
        data = sklearn.datasets.load_digits()
        x = data.data
        y = data.target
    
    elif name == 'MNIST':
        x,y = torch_to_numpy(torchvision.datasets.MNIST('./data'))
        
    elif name == 'FashionMNIST':
        x,y = torch_to_numpy(torchvision.datasets.FashionMNIST('./data')) 
        
    elif name == 'KMNIST':
        x,y = torch_to_numpy(torchvision.datasets.KMNIST('./data')) 

    else:
        print(f'there is no {name} dataset')
    
    return {'x':x,'y':y}
        
        