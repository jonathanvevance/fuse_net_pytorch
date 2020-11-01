import torch
import argparse
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

H = W = 32

class Hsigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace = self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction = 4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                                                                                                                             
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()                                                                                                                                                               
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)                                                                                                                                                     
        return x * y.expand_as(x)

class Hswish(nn.Module):
    def __init__(self, inplace = True):
        super(Hswish, self).__init__()                                                                                                                                                      
        self.inplace = inplace
                                                                                                                                                                                            
    def forward(self, x):
        return x * F.relu6(x + 3., inplace = self.inplace) / 6.

class FuseBlock(nn.Module):
    def __init__(self, inC, exp, K, stride, oup, is_SE, activ_func):

        super(FuseBlock, self).__init__()
        self.is_SE = is_SE
        self.activ_func = activ_func
        pad = (K - 1) // 2

        self.conv_11 = nn.Conv2d(
            in_channels = inC, out_channels = exp, kernel_size = 1, bias = False) 
        self.batch_norm_11 = nn.BatchNorm2d(num_features = exp)

        self.conv_dw_1k = nn.Conv2d(
            in_channels = exp, out_channels = exp, kernel_size = (1, K),
            groups = exp, stride = stride, padding = (0, pad), bias = False)
        self.batch_norm_1k = nn.BatchNorm2d(num_features = exp)

        self.conv_dw_k1 = nn.Conv2d(
            in_channels = exp, out_channels = exp, kernel_size = (K, 1),
            groups = exp, stride = stride, padding = (pad, 0), bias = False)
        self.batch_norm_k1 = nn.BatchNorm2d(num_features = exp)

        if self.is_SE:
            self.SEBlock = SEModule(2 * exp)
            self.HSLayer = Hsigmoid()

        self.conv_11_final = nn.Conv2d(
            in_channels = 2 * exp, out_channels = oup, kernel_size = 1, bias = False)
        self.batch_norm_11_final = nn.BatchNorm2d(num_features = oup)


    def forward(self, x):

        x = self.activ_func(self.batch_norm_11(self.conv_11(x)))
        left = self.batch_norm_1k(self.conv_dw_1k(x))
        right = self.batch_norm_k1(self.conv_dw_k1(x))
        x = torch.cat([left, right], 1)

        if self.is_SE:
            x = self.SEBlock(x)

        x = self.activ_func(x)
        x = self.batch_norm_11_final(self.conv_11_final(x))
        return x

class FuseNet(pl.LightningModule):
    def __init__(self):

        super(FuseNet, self).__init__()
        self.layers = nn.ModuleList()

        self.conv_1 = nn.Conv2d(
            in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.batch_norm_1c = nn.BatchNorm2d(num_features = 16)

        self.fuse_2 = FuseBlock(
            inC = 16, exp = 16, K = 3, stride = 2, oup = 16, is_SE = True, activ_func = nn.ReLU())

        self.fuse_3 = FuseBlock(
            inC = 16, exp = 72, K = 3, stride = 2, oup = 24, is_SE = False, activ_func = nn.ReLU())

        self.fuse_4 = FuseBlock(
            inC = 24, exp = 88, K = 3, stride = 1, oup = 24, is_SE = False, activ_func = nn.ReLU())
        self.layers.append(self.fuse_4)

        self.fuse_5 = FuseBlock(
            inC = 24, exp = 96, K = 5, stride = 2, oup = 40, is_SE = True, activ_func = Hswish())

        self.fuse_6 = FuseBlock(
            inC = 40, exp = 240, K = 5, stride = 1, oup = 40, is_SE = True, activ_func = Hswish())

        self.fuse_7 = FuseBlock(
            inC = 40, exp = 240, K = 5, stride = 1, oup = 40, is_SE = True, activ_func = Hswish())

        self.fuse_8 = FuseBlock(
            inC = 40, exp = 120, K = 5, stride = 1, oup = 48, is_SE = True, activ_func = Hswish())

        self.fuse_9 = FuseBlock(
            inC = 48, exp = 144, K = 5, stride = 1, oup = 48, is_SE = True, activ_func = Hswish())

        self.fuse_10 = FuseBlock(
            inC = 48, exp = 288, K = 5, stride = 2, oup = 96, is_SE = True, activ_func = Hswish())

        self.fuse_11 = FuseBlock(
            inC = 96, exp = 576, K = 5, stride = 1, oup = 96, is_SE = True, activ_func = Hswish())

        self.fuse_12 = FuseBlock(
            inC = 96, exp = 576, K = 5, stride = 1, oup = 96, is_SE = True, activ_func = Hswish())

        self.conv_2 = nn.Conv2d(
            in_channels = 96, out_channels = 576, kernel_size = 1, bias = False)
        self.batch_norm_2c = nn.BatchNorm2d(num_features = 576)

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size = 1)

        self.conv_3 = nn.Conv2d(
            in_channels = 576, out_channels = 1024, kernel_size = 1, bias = False)

        self.dropout = nn.Dropout(p = 0.2)

        self.conv_4 = nn.Conv2d(
            in_channels = 1024, out_channels = 100, kernel_size = 1, bias = False)

        self.layers = nn.Sequential(
            self.conv_1, self.batch_norm_1c, self.fuse_2, self.fuse_3, self.fuse_4,
            self.fuse_5, self.fuse_6, self.fuse_7, self.fuse_8, self.fuse_9, self.fuse_10, 
            self.fuse_11, self.fuse_12, self.conv_2, self.batch_norm_2c, Hswish(), 
            self.avg_pooling, self.conv_3, Hswish(), self.dropout, self.conv_4
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs).squeeze()
        loss = F.cross_entropy(output, labels)
        return {'loss': loss}

    def configure_optimizers(self):
        return optimizer

    def train_dataloader(self):
        
        train_set = CIFAR100(root = './data', train = True, 
                             download = True, transform = transform_train)

        train_loader = DataLoader(
            train_set, batch_size = args.batch_size, 
            shuffle = True, num_workers = 2)

        return train_loader

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs).squeeze()
        loss = F.cross_entropy(output, labels)
        return {'val_loss': loss}

    def val_dataloader(self):
        
        test_set = CIFAR100(root = './data', train = False, 
                             download = True, transform = transform_test)

        test_loader = DataLoader(
            test_set, batch_size = args.batch_size, 
            shuffle = False, num_workers = 2)

        return test_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

# transforms
transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),                                                                                                                               
                    transforms.RandomHorizontalFlip(),                                                                                                                                  
                    transforms.ToTensor(),                                                                                                                                              
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
])

transform_test = transforms.Compose([                                                                                                                                                   
                    transforms.ToTensor(),                                                                                                                                              
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                                                                                           
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help = "Enter epochs", type = int)
    parser.add_argument("--learning_rate", help = "Enter Learning Rate", type = float)
    parser.add_argument("--batch_size", help = "Enter Batch size", type = int)
    parser.add_argument("--optimizer", help = 'Enter optimizer (adam/sgd/rmsprop)')
    parser.add_argument("--betas_adam", help = 'Enter betas of adam (comma separated)', type = str)
    parser.add_argument("--mom_coeff", help = 'Enter coeff of momentum (float)', type = float)
    parser.add_argument("--alpha_rmsprop", help = 'Enter alpha of rmsprop (float)', type = float)
    parser.add_argument("--weight_decay", help = 'Enter weight decay coeff', type = float)

    args = parser.parse_args()
    assert args.epochs != None
    assert args.batch_size != None
    assert args.optimizer != None
    assert args.learning_rate != None

    model = FuseNet()

    # cmd line args
    global optimizer 
    if args.optimizer == 'adam':
        if args.betas_adam != None:
            beta1, beta2 = list(map(float, args.betas_adam.split(',')))
        else:
            beta1, beta2 = 0.9, 0.999

        if args.weight_decay != None:
            weight_decay = args.weight_decay

        optimizer = optim.Adam(
            model.parameters(), lr = args.learning_rate, betas = (beta1, beta2), weight_decay = weight_decay
        )

    elif args.optimizer == 'sgd':
        if args.mom_coeff != None:
            momentum = args.mom_coeff
        else:
            momentum = 0.9

        if args.weight_decay != None:
            weight_decay = args.weight_decay
        else:
            weight_decay = 0

        optimizer = optim.SGD(
            model.parameters(), lr = args.learning_rate, momentum = momentum, weight_decay = weight_decay, nesterov = True
        )

    elif args.optimizer == 'rmsprop':
        if args.alpha_rmsprop != None:
            alpha = args.alpha_rmsprop
        else:
            alpha = 0.99

        if args.weight_decay != None:
            weight_decay = args.weight_decay
        else:
            weight_decay = 0

        optimizer = optim.RMSprop(
            model.parameters(), lr = args.learning_rate, alpha = alpha, weight_decay = weight_decay
        )

    trainer = pl.Trainer(max_epochs = args.epochs, fast_dev_run = False, gpus = 1)
    trainer.fit(model)