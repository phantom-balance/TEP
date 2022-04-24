import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from seqloader import TEP
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Hyperparameters
feature_length = 52
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22
num_layers = 2
hidden_size = 40


# loading model
load_model = False
num_epochs = 10

# Tunings
sequence_length = 5
learning_rate = 0.001
batch_size = 10

print(f"seq_length{sequence_length}")
print(f"learning_rate{learning_rate}")
print(f"batch_size{batch_size}")
print(f"")
model = LSTM(feature_length=feature_length,sequence_length=sequence_length,hidden_size=hidden_size,num_layers=num_layers,num_classes=num_classes).to(device=device)

train_set = TEP(num=Type, sequence_length=sequence_length, is_train=True)
test_set = TEP(num=Type, sequence_length=sequence_length, is_train=False)

train_indices = pickle.load(open(f"processed_data/{sequence_length}-train_set_index.p", "rb"))
test_indices = pickle.load(open(f"processed_data/{sequence_length}-test_set_index.p", "rb"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)


def save_checkpoint(state, filename=f"model/LSTM_TEP_{sequence_length}.pth.tar"):
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


def loss_check(loader,model):
    loss_list=[]
    loss_av = 0
    loss_sum = 0
    with torch.no_grad():
        k = 0
        for Data, Targets in loader:
            Data = Data.to(device=device)
            Targets = Targets.to(device=device)
            scores = model(Data)
            loss = criterion(scores, Targets)
            loss_sum+=loss
            k+=1
        loss_av= loss_sum/k
        return loss_av

# To load the entire dataset:
# train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# To try out in a small dataset, for quick computation:
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(test_indices))



if load_model == True:
    load_checkpoint(torch.load(f"model/LSTM_TEP_{sequence_length}.pth.tar", map_location=device))


def train():
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs): # Here epoch doesn't mean going through the entire dataset
        for batch_idx, (data, targets) in enumerate(train_loader):
            # data, targets = next(iter(train_loader))
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            loss = criterion(scores,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # saving model after 2 epochs worth of training
        if epoch % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            print("Epoch no:",epoch)
            save_checkpoint(checkpoint)
            test_loss_temp=float(loss_check(test_loader,model))
            test_loss.append(test_loss_temp)
            train_loss_temp=float(loss_check(train_loader,model))
            train_loss.append(train_loss_temp)
            print("Train_Loss:",train_loss_temp)
            print("Test_Loss:",test_loss_temp)

    return train_loss, test_loss


if __name__ == "__main__":
  train_los, test_los = train()
  print(f"train-loss:{train_los}")
  print(f"test-loss:{test_los}")