import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from seqloader import TEP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRU(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.gru(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


input_size = 52
sequence_length = 5
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22
num_layers = 2
hidden_size = 40
learning_rate = 0.001
num_epochs = 100
batch_size = 50
load_model = True

model = GRU(input_size=input_size,sequence_length=sequence_length,hidden_size=hidden_size,num_layers=num_layers,num_classes=num_classes).to(device=device)

train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
test_set = TEP(num=Type, sequence_length=sequence_length, is_train=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),learning_rate)


def save_checkpoint(state, filename="model/GRU_TEP.pth.tar"):
    print("__Saving Checkpoint__")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


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
            num_correct += (prediction==Targets).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()


if __name__ == "__main__":

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    if load_model == True:
        load_checkpoint(torch.load("model/GRU_TEP.pth.tar"))

    for epoch in range(num_epochs): # Here epoch doesn't mean going through the entire dataset
        # for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = next(iter(train_loader))
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores,targets)
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
    Train_loader = DataLoader(dataset=train_set, batch_size=50, shuffle=False)
    Test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False)
    load_checkpoint(torch.load("model/GRU_TEP.pth.tar"))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(Train_loader):
                print(f'{100*(float(batch_idx)/(len(train_set)/batch_size)):.3f} completed')
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
            for batch_idx, (data, labels) in enumerate(Test_loader):
                print(f'{100*(float(batch_idx)/(len(test_set)/batch_size)):.3f} completed')
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
    else:
        print("enter either test or false")

    return y_true, y_pred, y_prob

# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)
# print("Checking accuracy on Testing Set")
# check_accuracy(test_loader, model)
