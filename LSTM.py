import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from seqloader import TEP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


input_size = 52
sequence_length = 5
Type = [1,2]
num_classes = 22
num_layers = 2
hidden_size = 30
learning_rate = 0.001
num_epochs = 2
batch_size = 10
load_model = False

model=LSTM(input_size=input_size,sequence_length=sequence_length,hidden_size=hidden_size,num_layers=num_layers,num_classes=num_classes).to(device=device)

train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# print("train_set",train_set[476])
test_set = TEP(num=Type, sequence_length=sequence_length, is_train=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
# print("test_set",test_set[1112])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rate)


def save_checkpoint(state, filename="LSTM_TEP.pth.tar"):
    print("__Saving Checkpoint__")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if load_model == True:
    load_checkpoint(torch.load("LSTM_TEP.pth.tar"))


def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for Data, Targets in loader:
            Data = Data.to(device=device).squeeze(1)
            Targets = Targets.to(device=device)
            scores = model(Data)

            _, prediction = scores.max(1)
            num_correct += (prediction==Targets).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()



for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        save_checkpoint(checkpoint)
        print("Checking accuracy on Training Set")
        check_accuracy(train_loader, model)
        print("Checking accuracy on Testing Set")
        check_accuracy(test_loader, model)




