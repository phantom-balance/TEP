import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from seqloader import TEP
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature_length = 52
sequence_length = 5
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22
num_layers = 2
hidden_size = 40
learning_rate = 0.001
num_epochs = 0
batch_size = 5
load_model = True
small_data_size = 5


train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
small_train_set, _ = random_split(train_set, [small_data_size, len(train_set)-small_data_size])
test_set = TEP(num=Type, sequence_length=sequence_length, is_train=False)
small_test_set, _ = random_split(test_set, [small_data_size, len(test_set)-small_data_size])

train_loader = DataLoader(dataset=small_train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=small_test_set, batch_size=batch_size, shuffle=True)


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
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


class RNN(nn.Module):
    def __init__(self, feature_length, sequence_length, hidden_size, num_layers, num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(feature_length, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


rnn_model = RNN(feature_length=feature_length, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)


def rnn_load_checkpoint(checkpoint):
    print("__Loading RNN Checkpoint__")
    rnn_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def rnn_summary_return(DATA):

    rnn_load_checkpoint(torch.load("model/RNN_TEP.pth.tar", map_location=device))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        print("Checking accuracy on Training Set")
        check_accuracy(train_loader, rnn_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = rnn_model(data)
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
        print("Checking accuracy on Testing Set")
        check_accuracy(test_loader, rnn_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = rnn_model(data)
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


class LSTM(nn.Module):
    def __init__(self, feature_length, sequence_length, hidden_size, num_layers, num_classes):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_length, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.lstm(x,(h0,c0))
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


lstm_model = LSTM(feature_length=feature_length, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)


def lstm_load_checkpoint(checkpoint):
    print("__Loading LSTM Checkpoint__")
    lstm_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def lstm_summary_return(DATA):

    lstm_load_checkpoint(torch.load("model/LSTM_TEP.pth.tar", map_location=device))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        print("Checking accuracy on Training Set")
        check_accuracy(train_loader, lstm_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = lstm_model(data)
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
        print("Checking accuracy on Testing Set")
        check_accuracy(test_loader, lstm_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = lstm_model(data)
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


class GRU(nn.Module):
    def __init__(self, feature_length, sequence_length, hidden_size, num_layers, num_classes):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(feature_length, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.gru(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


gru_model = GRU(feature_length=feature_length, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)


def gru_load_checkpoint(checkpoint):
    print("__Loading GRU Checkpoint__")
    gru_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def gru_summary_return(DATA):

    gru_load_checkpoint(torch.load("model/GRU_TEP.pth.tar", map_location=device))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        print("Checking accuracy on Training Set")
        check_accuracy(train_loader, gru_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = gru_model(data)
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
        print("Checking accuracy on Testing Set")
        check_accuracy(test_loader, gru_model)
        print('computing......')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                data = data.to(device=device)
                labels = labels.to(device=device)
                scores = gru_model(data)
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


if __name__ == "__main__":
    rnn_load_checkpoint(torch.load("model/RNN_TEP.pth.tar", map_location=device))
    print(rnn_model)
    print("Number of parameters: ", sum(p.numel() for p in rnn_model.parameters()))
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, rnn_model)
    print("Checking accuracy on Testing Set")
    check_accuracy(test_loader, rnn_model)

    lstm_load_checkpoint(torch.load("model/LSTM_TEP.pth.tar", map_location=device))
    print(lstm_model)
    print("Number of parameters: ", sum(p.numel() for p in lstm_model.parameters()))
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, lstm_model)
    print("Checking accuracy on Testing Set")
    check_accuracy(test_loader, lstm_model)

    gru_load_checkpoint(torch.load("model/GRU_TEP.pth.tar", map_location=device))
    print(gru_model)
    print("Number of parameters: ", sum(p.numel() for p in gru_model.parameters()))
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, gru_model)
    print("Checking accuracy on Testing Set")
    check_accuracy(test_loader, gru_model)
