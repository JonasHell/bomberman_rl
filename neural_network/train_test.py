import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from neural_network.dataset import BomberManDataSet
from neural_network.neural_network import OurNeuralNetwork


# hyperparameters
input_size = 257
num_of_epochs = 20
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name = "7x7_ep"+str(num_of_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)

# data sets and data loaders
train_set = BomberManDataSet("training_data/", "coin")
test_set = BomberManDataSet("test_data/", "coin")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# summary writer for tensorboard
writer = SummaryWriter("runs/" + name)

# load model or initialize new one
if os.path.isfile(name+".pt"):
    model = torch.load(name+".pt")
    model = model.to(device)
else:
    model = OurNeuralNetwork(input_size).to(device)

# define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# initialize metrics
running_loss_train = 0
running_correct_train = 0

# set other variables
num_total_steps_train = len(train_loader)
num_total_steps_test = len(test_loader)

# learning process
for epoch in range(num_of_epochs):

    # set model to training mode
    model.train()

    # training loop
    for batch, (features, labels) in enumerate(train_loader):

        # push everything to device
        features = features.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(features)
        loss = criterion(output, labels)

        # clear previous gradients
        model.zero_grad()

        # backward pass
        loss.backward()
        optimizer.step()

        # update metrics
        running_loss_train += loss.item()
        _, predicted = torch.max(output, 1)
        correct = (predicted == labels).sum().item()
        running_correct_train += correct

        # print and tensorboard output
        if (batch+1) % 100 == 0:
            print(f"[{epoch+1}/{num_of_epochs}] [{batch+1}/{num_total_steps_train}] loss={loss.item():.4f}, acc={correct*100./batch_size}%")
            writer.add_scalar('training loss', running_loss_train/100, epoch*num_total_steps_train+batch)
            writer.add_scalar('training accuracy', running_correct_train/100, epoch*num_total_steps_train+batch)

            # set matrics back to zero
            running_loss_train = 0
            running_correct_train = 0


    # init metrics for testing
    running_loss_test = 0
    running_correct_test = 0

    # set model to eval mode
    model.eval()

    # turn off gradient calculation for testing
    with torch.no_grad():

        # testing loop
        for batch, (features, labels) in enumerate(test_loader):

            # push everything to device
            features = features.to(device)
            labels = labels.to(device)

            # forward pass
            output = model(features)
            loss = criterion(output, labels)

            # update metrics
            running_loss_test += loss.item()
            _, predicted = torch.max(output, 1)
            running_correct_test += (predicted == labels).sum().item()

        # print and tensorboard
        print(f"test: [{epoch+1}/{num_of_epochs}] loss={running_loss_test/num_total_steps_test:.4f}, acc={running_correct_test*100./num_total_steps_test/batch_size}%")
        writer.add_scalar('training loss', running_loss_train/num_total_steps_test, epoch)
        writer.add_scalar('training accuracy', running_correct_train/num_total_steps_test/batch_size, epoch)

        # set matrics back to zero
        running_loss_train = 0
        running_correct_train = 0

    # save model
    torch.save(model, name+".pt")
