# import torch and other necessary modules from torch
import torch
import torch.nn as nn
...
# import torchvision and other necessary modules from torchvision 
import torchvision
...
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score

# recommended preprocessing steps: resize to square -> convert to tensor -> normalize the image
# if you are resizing, 100 is a good choice otherwise GradeScope will time out
# you could use Compose (https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) from transforms module to handle preprocessing more conveniently
# transform = transforms.Compose(...)

transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the image to a 100x100 square
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])


# thanks to torchvision, this is a convenient way to read images from folders directly without writing datasets class yourself (you should know what datasets class is as mentioned in the documentation)
# dataset = datasets.ImageFolder(...)

dataset = datasets.ImageFolder(root = './petimages', transform = transform)


# now we need to split the data into training set and evaluation set 
# use 20% of the dataset as test
# test_set, train_set = torch.utils.data.random_split(...)

test_set, train_set = torch.utils.data.random_split(dataset, [int(0.2 * len(dataset)), len(dataset) - int(0.2 * len(dataset))])

# model hyperparameter
learning_rate = 0.001
batch_size = 32
epoch_size = 10

# test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
# train_set = torch.utils.data.Subset(dataset, range(n_test, len(dataset)))  # take the rest  
# trainloader = torch.utils.data.DataLoader(...)
# testloader = ...

trainloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# prepare dataloader for training set and evaluation set
# trainloader = torch.utils.data.DataLoader(...)
# testloader = ...

trainloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)


# model design goes here
class CNN(nn.Module):

    # there is no "correct" CNN model architecture for this lab, you can start with a naive model as follows:
    # convolution -> relu -> pool -> convolution -> relu -> pool -> convolution -> relu -> pool -> linear -> relu -> linear -> relu -> linear
    # you can try increasing number of convolution layers or try totally different model design
    # convolution: nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    # pool: nn.MaxPool2d (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    # linear: nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    def __init__(self):
        super(CNN,self).__init__()
        # ...
        self.convolution_one = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.relu_one = nn.ReLU()
        self.pool_one = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convolution_two = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu_two = nn.ReLU()
        self.pool_two = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convolution_three = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu_three = nn.ReLU()
        self.pool_three = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_one = nn.Linear(64 * 12 * 12, 128)

        self.relu_four = nn.ReLU()
        
        self.fc_two = nn.Linear(128, 64)

        self.relu_five = nn.ReLU()
        
        self.fc_three = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool_one(self.relu_one(self.convolution_one(x)))
        x = self.pool_two(self.relu_two(self.convolution_two(x)))
        x = self.pool_three(self.relu_three(self.convolution_three(x)))
        
        x = x.view(x.size(0), -1)

        x = self.relu_four(self.fc_one(x))
        x = self.relu_five(self.fc_two(x))

        x = self.fc_three(x)
        
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU
# search in official website for CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)


# start model training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size): # begin with trying 10 epochs 

    loss = 0.0 # you can print out average loss per batch every certain batches

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, label = data
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        label = label.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs = cnn(inputs)
        loss = criterion(outputs, label)
        loss.backward() 
        optimizer.step()

        # print some statistics
        loss += loss.item() # add loss for current batch 
        if i % 100 == 99:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() # turn on evaluation model, also a good practice to do
with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs, so turn on no_grad mode
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels.tolist() # convert labels to list and append to ground_truth
        # calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        # the class with the highest logit is what we choose as prediction
        # _, predicted = torch.max(...)
        _, predicted = torch.max(outputs, 1)
        # prediction += ... # convert predicted to list and append to prediction
        prediction += predicted.tolist()

# GradeScope is chekcing for these three variables, you can use sklearn to calculate the scores
# accuracy = accuracy_score(...)
# recall = recall_score(...)
# precision = precision_score(...)

accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='weighted') 
precision = precision_score(ground_truth, prediction, average='weighted')






