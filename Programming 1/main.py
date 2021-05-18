from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar
import dla

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

def default_loader(path):
    return Image.open(path).convert('RGB')

'''定义数据集'''
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if(txt!="./data/test.txt"):
                imgs.append((words[0], int(words[1])))
            else:
                imgs.append((words[0], -1))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=6),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize(32),
    # transforms.Normalize((0.47659552,0.45051836,0.40219973), (0.2260658,0.22366214,0.22351762))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32)
])

train_data = MyDataset('./data/train.txt', transform=transform_train)
val_data = MyDataset('./data/val.txt', transform=transform_test)
test_data = MyDataset('./data/test.txt', transform=transform_test)

trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
valloader = DataLoader(dataset=val_data, batch_size=64)
testloader = DataLoader(dataset=test_data, batch_size=1)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
           '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
           '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54',
           '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
           '73', '74', '75', '76', '77', '78', '79',)

print('==> Building model..')

net = dla.DLA()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Best acc so far: ' + str(acc))
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_resnet.pth')
        best_acc = acc

        with open('./result/test_resnet.txt', "w") as f:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                f.write(str(batch_idx)+'.jpg'+' '+str(predicted.numpy()[0]))
                f.write('\n')

for epoch in range(start_epoch, start_epoch+5000):
    train(epoch)
    test(epoch)
    scheduler.step()