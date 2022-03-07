# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loader import TEP

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 52
Type = [0, 1]
num_classes = 2 # requires to be converted to num_classes = len(Type) to change according to the number of files sent to training and testing
learning_rate = 0.001
num_epochs = 3
batch_size = 13
load_model = True


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

# Load Data
train_set = TEP(num=Type, is_train=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = TEP(num=Type, is_train=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Saving and Loading the model parameters
def save_checkpoint(state, filename="NN_TEP.pth.tar"):
    print("__Saving Checkpoint__")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if load_model == True:
    load_checkpoint(torch.load("NN_TEP.pth.tar"))

# Training Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    if epoch % 2 == 0:
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        save_checkpoint(checkpoint)


# Testing accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for Data, Targets in loader:
            Data = Data.to(device=device)
            Targets = Targets.to(device=device)
            scores = model(Data)
            _, predictions = scores.max(1)
            num_correct += (predictions==Targets).sum()
            num_samples += predictions.size(0)
        print(f'In training dataset Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)
print("Checking accuracy on Testing Set")
check_accuracy(test_loader, model)
