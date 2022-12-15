down load CIFAR-10 data using follow codes.

```py
trainset = torch_datasets.CIFAR10(root='./Data', train=True, download=True)
  testset = torch_datasets.CIFAR10(root='./Data', train=False, download=True)
```
