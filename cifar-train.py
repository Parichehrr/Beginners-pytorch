from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import argparse
import torch.optim as optim
parser = argparse.ArgumentParser(description='PyTorch cifar Example')

parser.add_argument('--batch-size', type=int, default=4, metavar='N',

                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',

                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',

                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',

                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',

                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,

                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',

                    help='random seed (default: 1)')

parser.add_argument('--log', type=int, default=2000, metavar='N',

                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)

if args.cuda:

    torch.cuda.manual_seed(args.seed)



kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# A simple CNN with 6 layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1000, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1000, 1300, 3,stride=2)
        self.fc1 = nn.Linear(1300 * 3*3, 784)
        self.fc2 = nn.Linear(784, 120)
        self.fc3 = nn.Linear(120, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1300*3*3)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net=Net()

if args.cuda:
    net.cuda()

loss_fun = nn.CrossEntropyLoss()

optimizer=optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)

for epoch in range(args.epochs):

    running_loss=0.0

    for i,data in enumerate(train_loader,0):
        inputs,labels=data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs=net(inputs)

        loss=loss_fun(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.data[0]

        if i % args.log==1999:

             print('[%d, %5d] loss: %.3f' %

                     (epoch + 1, i + 1, running_loss / args.log))

             running_loss = 0.0

    print("finished Epoch {}".format(epoch+1))

correct = 0
total = 0

for data,target in test_loader:

    if args.cuda:

            data, target = data.cuda(), target.cuda()

    data, target = Variable(data, volatile=True), Variable(target)
    output=net(data)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)

    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

    correct += pred.eq(target.data.view_as(pred)).cpu().sum()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))
