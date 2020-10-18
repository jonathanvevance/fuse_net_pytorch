import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
print(torch.cuda.get_device_name(0))

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help = "Enter epochs", type = int)
parser.add_argument("--learning_rate", help = "Enter Learning Rate", type = float)
parser.add_argument("--batch_size", help = "Enter Batch size", type = int)
parser.add_argument("--optimizer", help = 'Enter optimizer (adam/sgd/rmsprop)')
parser.add_argument("--betas_adam", help = 'Enter betas of adam (comma separated)', type = str)
parser.add_argument("--mom_coeff", help = 'Enter coeff of momentum (float)', type = float)
parser.add_argument("--alpha_rmsprop", help = 'Enter alpha of rmsprop (float)', type = float)
parser.add_argument("--scheduler", help = 'Enter True/False', type = bool)

args = parser.parse_args()
assert args.epochs != None
assert args.batch_size != None
assert args.optimizer != None
assert args.learning_rate != None

wandb.login(key = 'b567ff0e49926099eea499997b7a78c48d2bbf48')
wandb.init(project = 'fusenet')

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

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
        left = self.batch_norm_1k(self.conv_dw_1k(x))  ## try same batch norm
        right = self.batch_norm_k1(self.conv_dw_k1(x)) ## try same batch norm
        x = torch.cat([left, right], 1)

        if self.is_SE:
            # x = self.HSLayer(self.SEBlock(x))
            x = self.SEBlock(x) # ask TA

        x = self.activ_func(x)
        x = self.batch_norm_11_final(self.conv_11_final(x))
        return x

class FuseNet(nn.Module):
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

def initialize_weights(net):

    for m in net.modules():                                                                                          
        if isinstance(m, nn.Conv2d):                                                                                         
            nn.init.kaiming_normal_(m.weight, mode='fan_out')                                                                
            if m.bias is not None:                                                                                           
                nn.init.zeros_(m.bias)                                                                                       
        elif isinstance(m, nn.BatchNorm2d):                                                                                  
            nn.init.ones_(m.weight)                                                                                          
            nn.init.zeros_(m.bias)                                                                                           
        elif isinstance(m, nn.Linear):                                                                                       
            nn.init.normal_(m.weight, 0, 0.01)                                                                               
            if m.bias is not None:                                                                                           
                nn.init.zeros_(m.bias)

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

train_set = CIFAR100(root = './data', train = True, download = True, transform = transform_train)
test_set = CIFAR100(root = './data', train = False, download = True, transform = transform_test)

train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 2)
test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = 2)

model = FuseNet().to(device)
model.apply(initialize_weights)
criterion = nn.CrossEntropyLoss()

# parsing cmd line args
if args.optimizer == 'adam':
    if args.betas_adam != None:
        beta1, beta2 = list(map(float, args.betas_adam.split(',')))
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, betas = (beta1, beta2))
    else:
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

elif args.optimizer == 'sgd':
    if args.mom_coeff != None:
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.mom_coeff, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, nesterov = True)

elif args.optimizer == 'rmsprop':
    if args.alpha_rmsprop != None:
        optimizer = optim.RMSprop(model.parameters(), lr = args.learning_rate, alpha = args.alpha_rmsprop)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr = args.learning_rate)

scheduler = None
if args.scheduler != None:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

wandb.watch(model, log = 'all')

def train():

    best_test_acc = 0.0
    running_train_acc = 0.0
    running_train_loss = 0.0

    for epoch in range(args.epochs):
        print(f'\nepoch {epoch}')
        for i, (inputs, labels) in enumerate(train_loader, 0):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs).squeeze()
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            output = model(inputs).squeeze()
            preds = torch.argmax(output, dim = 1)
            accuracy = (preds == labels).float().mean()
            running_train_acc += accuracy
            running_train_loss += loss.item()

            print(f"\rloss: {loss.item()}, accuracy = {accuracy}", end = "", flush = True)

            if (i % 2000 == 1999) and (i > 0):

                wandb.log({
                    "running_acc_2000" : running_acc / 2000,
                    "running_loss_2000" : running_loss/2000
                })

                running_train_acc = running_train_loss = 0

            if scheduler != None:
                scheduler.step()

        with torch.no_grad():
            avg_train_loss, avg_train_acc = [], []
            for j, (input_train, y_train) in enumerate(train_loader):
                input_train, y_train = input_train.to(device), y_train.to(device)
                train_preds = model(input_train).squeeze()
                train_loss = criterion(train_preds, y_train)
                avg_train_loss.append(train_loss.item())
                train_preds = torch.argmax(train_preds, dim = 1)
                train_acc = (train_preds == y_train).float().mean().item()
                avg_train_acc.append(train_acc)

                if j == 14:
                    break

            avg_train_loss = sum(avg_train_loss) / len(avg_train_loss)
            avg_train_acc = sum(avg_train_acc) / len(avg_train_acc)
            print(f"\nTrain loss = {round(avg_train_loss, 4)}; Train accuracy = {round(avg_train_acc, 4)} (on {15*args.batch_size} imgs)")

            avg_test_loss, avg_test_acc = [], []
            for j, (input_test, y_test) in enumerate(test_loader):
                input_test, y_test = input_test.to(device), y_test.to(device)
                test_preds = model(input_test).squeeze()
                test_loss = criterion(test_preds, y_test)
                avg_test_loss.append(test_loss.item())
                test_preds = torch.argmax(test_preds, dim = 1)
                test_acc = (test_preds == y_test).float().mean().item()
                avg_test_acc.append(test_acc)

            avg_test_loss = sum(avg_test_loss) / len(avg_test_loss)
            avg_test_acc = sum(avg_test_acc) / len(avg_test_acc)
            print(f"Test loss = {round(avg_test_loss, 4)}; Test accuracy = {round(avg_test_acc, 4)}\n")

            wandb.log({
                "test_acc": avg_test_acc,
                "test_loss": avg_test_loss,
                "train_acc_mini": avg_train_acc,
                "train_loss_mini": avg_train_loss
            })

            if avg_test_acc > best_test_acc:
                wandb.save('best' + str(avg_test_acc) + '.h5')
                best_test_acc = avg_test_acc
            else:
                wandb.save('latest' + str(avg_test_acc) + '.h5')

    print('\n\nFinished Training')
    with torch.no_grad():
        avg_test_loss, avg_test_acc = [], []
        for j, (input_test, y_test) in enumerate(test_loader):
            input_test, y_test = input_test.to(device), y_test.to(device)
            test_preds = model(input_test).squeeze()
            test_loss = criterion(test_preds, y_test)
            avg_test_loss.append(test_loss.item())
            test_preds = torch.argmax(test_preds, dim = 1)
            test_acc = (test_preds == y_test).float().mean().item()
            avg_test_acc.append(test_acc)

        avg_test_loss = sum(avg_test_loss) / len(avg_test_loss)
        avg_test_acc = sum(avg_test_acc) / len(avg_test_acc)
        print(f"Final test loss = {round(avg_test_loss, 4)}; Final test accuracy = {round(avg_test_acc, 4)}\n")
        wandb.log({
            'final_test_acc' : avg_test_acc,
            'final_test_loss' : avg_test_loss
        })

train()