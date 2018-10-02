import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch


class Trainer:
    def __init__(self, model):
        self.model = model
        self.logger = SummaryWriter()

    def train(self, train_loader, test_loader, criterion=nn.CrossEntropyLoss(), optimizer=None, epoch_cnt=50):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epoch_cnt):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.logger.add_scalar('train loss', self.calc_loss(train_loader, criterion), epoch)
            self.logger.add_scalar('test loss', self.calc_loss(test_loader, criterion), epoch)
            self.logger.add_scalar('test accuracy', self.calc_accuracy(test_loader), epoch)

    def calc_accuracy(self, data_loader):
        correct = 0
        total = 0
        for inputs, labels in data_loader:
            outputs = self.model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def calc_loss(self, data_loader, criterion):
        loss = 0
        for inputs, labels in data_loader:
            outputs = self.model.forward(inputs)
            loss += criterion(outputs, labels).item()
        return loss
