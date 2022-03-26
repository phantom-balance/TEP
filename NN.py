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
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22
learning_rate = 0.001
num_epochs = 50
batch_size = 10
load_model = False


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,30)
        self.fc2 = nn.Linear(30, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# main data
train_set = TEP(num=Type, is_train=True)
test_set = TEP(num=Type, is_train=False)

model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

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


if __name__ == '__main__':
    # Load Data
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    if load_model == True:
        load_checkpoint(torch.load("NN_TEP.pth.tar"))

    # Training Network
    for epoch in range(num_epochs):
        # for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = next(iter(train_loader))
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint)


# for performance_metric
def summary_return(DATA):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Train_loader = DataLoader(dataset=train_set, batch_size=50, shuffle=False)
    Test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False)
    load_checkpoint(torch.load("NN_TEP.pth.tar"))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(Test_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = model(data)
                prob = nn.Softmax(dim=1)
                y_prob_temp = prob(scores)
                if batch_idx == 0:
                    y_prob = y_prob_temp
                else:
                    y_prob = torch.cat((y_prob, y_prob_temp), dim=0)
                _, predictions = scores.max(1)
                y_pred.extend(predictions)
                y_true.extend(labels)
    elif DATA == "test":
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(Train_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = model(data)
                prob = nn.Softmax(dim=1)
                y_prob_temp = prob(scores)
                if batch_idx == 0:
                    y_prob = y_prob_temp
                else:
                    y_prob = torch.cat((y_prob, y_prob_temp), dim=0)
                _, predictions = scores.max(1)
                y_pred.extend(predictions)
                y_true.extend(labels)

    return y_true, y_pred, y_prob

# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)
# print("Checking accuracy on Testing Set")
# check_accuracy(test_loader, model)
