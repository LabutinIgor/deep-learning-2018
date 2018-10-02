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


def test_save_load():
    model = resnext50(num_classes=10)
    trainer = Trainer(model)
    train_set = TensorDataset(torch.rand(2, 3, 224, 224), torch.randint(0, 10, (2,)).long())
    test_set = TensorDataset(torch.rand(2, 3, 224, 224), torch.randint(0, 10, (2,)).long())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4)
    trainer.train(train_loader, test_loader, epoch_cnt=1)
    trainer.save_weights('weights.txt')
    model2 = resnext50(pretrained_weights=torch.load('weights.txt'), num_classes=10)

    for i, j in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.all(torch.eq(i[1], j[1]))
