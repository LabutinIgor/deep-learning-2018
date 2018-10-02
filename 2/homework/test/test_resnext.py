from resnext.resnext import *
from resnext.train import *
import torch
from torch.utils.data import TensorDataset


def test_resnext():
    model = resnext50(num_classes=10)
    trainer = Trainer(model)
    train_set = TensorDataset(torch.rand(5, 3, 224, 224), torch.randint(0, 10, (5,)).long())
    test_set = TensorDataset(torch.rand(5, 3, 224, 224), torch.randint(0, 10, (5,)).long())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4)
    trainer.train(train_loader, test_loader, epoch_cnt=2)
