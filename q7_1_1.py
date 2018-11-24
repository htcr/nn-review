import torch
from hw5datasets import NIST36Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
from hw5models import NaiveNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16
data_format = 'vector'

train_set = NIST36Dataset(partition='train', data_format=data_format)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

valid_set = NIST36Dataset(partition='valid', data_format=data_format)
valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, num_workers=1)

test_set = NIST36Dataset(partition='test', data_format=data_format)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=1)

net = NaiveNet()
net = net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

max_epochs = 100

best_model = copy.deepcopy(net.state_dict())
best_acc = 0.0

losses = list()
epochs = list()
train_accs = list()
val_accs = list()

for epoch in range(max_epochs):
    print('Epoch {}/{}'.format(epoch, max_epochs - 1))
    net.train()

    train_loss = 0.0
    train_correct = 0

    torch.set_grad_enabled(True)

    for idx, batch in enumerate(train_loader):
        samples, labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = net(samples)

        _, preds = torch.max(outputs, 1)
        batch_loss = loss_func(outputs, labels)
        batch_loss.backward()

        optimizer.step()

        train_loss += batch_loss.item() * samples.shape[0]
        train_correct += torch.sum(preds == labels)

    train_loss = train_loss / len(train_set)
    train_acc = float(train_correct) / len(train_set)
    
    print('Train Loss: {} Train Acc: {:.2f}'.format(train_loss, train_acc))

    val_correct = 0
    net.eval()
    torch.set_grad_enabled(False)

    for samples, labels in valid_loader:
        samples = samples.to(device)
        labels = labels.to(device)

        outputs = net(samples)
        _, preds = torch.max(outputs, 1)

        val_correct += torch.sum((preds == labels))

    val_acc = float(val_correct) / len(valid_set)
    
    print('Val Acc: {:.2f}'.format(val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(net.state_dict())

    epochs.append(epoch)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    losses.append(train_loss)

torch.save(best_model, 'best_model_naive_net.pth')

net.load_state_dict(best_model)
test_correct = 0
net.eval()
torch.set_grad_enabled(False)

for samples, labels in test_loader:
    samples = samples.to(device)
    labels = labels.to(device)

    outputs = net(samples)
    _, preds = torch.max(outputs, 1)

    test_correct += torch.sum((preds == labels))

test_acc = float(test_correct) / len(test_set)

print('Test Acc: {:.2f}'.format(test_acc))