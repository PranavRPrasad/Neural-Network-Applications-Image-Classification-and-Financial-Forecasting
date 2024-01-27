# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score

# Create the dataframe by reading the dataset
# rnn_df = pd.read_csv('coin_Bitcoin.csv')
rnn_df = pd.read_csv('./coin_Bitcoin.csv')

# take a look at the csv file yourself first
# columns High, Low, Open are input features and column Close is target value
x = rnn_df[['High', 'Low', 'Open']].values
y = rnn_df['Close'].values 

# use StandardScaler from sklearn to standardize
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# split into train and evaluation (8 : 2) using train_test_split from sklearn
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 50)


# now make x and y tensors, think about the shape of train_x, it should be (total_examples, sequence_lenth, feature_size)
# we wlll make sequence_length just 1 for simplicity, and you could use unsqueeze at dimension 1 to do this
# also when you create tensor, it needs to be float type since pytorch training do not take default type read using pandas
train_x = torch.tensor(train_x, dtype = torch.float32)
train_y = torch.tensor(train_y, dtype = torch.float32)
test_x = torch.tensor(test_x, dtype = torch.float32)

train_x = train_x.unsqueeze(1)
seq_len = train_x[0].shape[0] # it is actually just 1 as explained above


# different from CNN which uses ImageFolder method, we don't have such method for RNN, so we need to write dataset class ourselves, reference tutorial is in main documentation
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(BitCoinDataSet, self).__init__()
        self.data = train_x
        self.labels = train_y
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# now prepare dataloader for training set and evaluation set, and hyperparameters
hidden_size = 64
num_layers = 2
learning_rate = 0.001
batch_size = 32
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


# model design goes here
class RNN(nn.Module):

    # there is no "correct" RNN model architecture for this lab either, you can start with a naive model as follows:
    # lstm with 5 layers (or rnn, or gru) -> linear -> relu -> linear
    # lstm: nn.LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.fc(output)
        return output


# instantiate your rnn model and move to device as in cnn section
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rnn = RNN(input_feature_size = 3, hidden_size = hidden_size, num_layers = num_layers).to(device)
# loss function is nn.MSELoss since it is regression task
criteria = nn.MSELoss()
# yo ucan start with using Adam as optimizer as well 
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)



# start training 
rnn.train()
for epoch in range(epoch_size): # start with 10 epochs

    loss = 0.0 # you can print out average loss per batch every certain batches

    for batch_idx, data in enumerate(train_loader):
        # get inputs and target values from dataloaders and move to device
        inputs, targets = data
        inputs.to(device)
        targets.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs = rnn(inputs)
        loss = criteria(outputs, targets)
        loss.backward()
        optimizer.step()

        loss += loss.item() # add loss for current batch
        if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')



prediction = []
ground_truth = []
# evaluation
rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        inputs = inputs.to(device)

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out


# remember we standardized the y value before, so we must reverse the normalization before we compute r2score
prediction = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1))
ground_truth = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1))

# use r2_score from sklearn
r2score = r2_score(prediction,ground_truth)