import torch
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt
import torchvision.models as models
from hw5models import LeNetBNRGB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data_transform = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_data_transform = transforms.Compose([
        transforms.Resize(30),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_set = datasets.ImageFolder(
    root='../data/oxford-flowers17/train',
    transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=4)

valid_set = datasets.ImageFolder(
    root='../data/oxford-flowers17/val',
    transform=test_data_transform)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=4, shuffle=False, num_workers=4)

test_set = datasets.ImageFolder(
    root='../data/oxford-flowers17/test',
    transform=test_data_transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False, num_workers=4)


net = LeNetBNRGB(num_class=17)
net = net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
step_scheduler = lr_scheduler.MultiStepLR(optimizer, [100], gamma=0.1)

max_epochs = 60

best_model = copy.deepcopy(net.state_dict())
best_acc = 0.0

losses = list()
epochs = list()
train_accs = list()
val_accs = list()

for epoch in range(max_epochs):
    print('Epoch {}/{}'.format(epoch, max_epochs - 1))
    step_scheduler.step()
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

torch.save(best_model, 'best_model_flower_lenet.pth')

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

plt.subplot(2, 1, 1)
plt.plot(epochs, train_accs)
plt.plot(epochs, val_accs)
plt.ylabel('train/val accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs, losses)
plt.ylabel('loss')

plt.xlabel('epochs')

plt.show()