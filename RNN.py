import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader2 import TEP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _  = self.rnn(x,h0)
        print(out.shape)
        out = out.fc(out)
        return out

#print(RNN(input_size=52, sequence_length=10, hidden_size=100, num_layers=2, output_size=22))

# Hyperparameters
input_size = 52
sequence_length = 0
output_size = 22
num_layers = 2
hidden_size = 200
learning_rate = 0.001
batch_size = 10
num_epochs = 1
Type = [0, 1]

# Initializing the network
model = RNN(input_size=input_size, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device=device)

# Loading data
train_set = TEP(num=Type, is_train=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = TEP(num=Type, is_train=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Optimizre and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Training the network
for epochs in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        print(data.shape)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Evaluation
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for Data, Targets in loader:
            Data = Data.to(device=device)
            Targets = Targets.to(device=device)
            scores = model(Data)

            _, prediction = scores.max(1)
            num_correct+=(prediction==Targets).sum()
            num_samples+=prediction.size(0)

        print(f'Got{num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)
print("Checking accuracy on Testing Set")
check_accuracy(test_loader, model)
