#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import dataset
import modules
import matplotlib.pyplot as plt

# Meta-parameters
n_iterations = 1000
learning_rate = 0.001
dropout = False

# Training dataset loader
training_dataset = dataset.SharedTaskDataset('./data/cwi-train-cx-1.csv', train=True)
trainloader = DataLoader(training_dataset, batch_size=64, shuffle=False, num_workers=2)

# Test dataset loader
test_dataset = dataset.SharedTaskDataset('./data/cwi-train-cx-1.csv', train=False)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Classes
classes = ('non-complex', 'complex')

# CNN classifier
cnn_classifier = modules.CNNClassifier(dropout=dropout)
cnn_classifier.cuda()

# SGD optimizer
optimizer = optim.SGD(cnn_classifier.parameters(), lr=learning_rate, momentum=0.9)

# Objective function
criterion = nn.CrossEntropyLoss()

# Losses and accuracies
train_losses = torch.zeros(n_iterations)
train_accuracies = torch.zeros(n_iterations)
test_losses = torch.zeros(n_iterations)
test_accuracies = torch.zeros(n_iterations)

# For each iteration
for epoch in range(n_iterations):
    # Data to compute accuracy
    total = 0.0
    success = 0.0

    # For each batch
    for data in trainloader:
        # Get inputs and labels
        inputs, labels = data

        # To variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # Gradients to zero
        optimizer.zero_grad()

        # Forward and loss
        outputs = cnn_classifier(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Take the max as prediction
        _, predicted = torch.max(outputs.data, 1)

        # Add to total
        total += labels.size(0)

        # Add correctly classified sample
        success += (predicted == labels.data).sum()
    # end for

    # Print and save loss
    print(u"Train Epoch {} Loss : {} Accuracy : {}".format(epoch, float(loss.data), success / total))
    train_losses[epoch] = float(loss.data)
    train_accuracies[epoch] = success / total

    # Test data
    total = 0.0
    success = 0.0

    # On the test set
    for data in testloader:
        # Get inputs and labels
        inputs, labels = data

        # To variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # Forward
        outputs = cnn_classifier(inputs)
        loss = criterion(outputs, labels)

        # Take the max is predicted
        _, predicted = torch.max(outputs.data, 1)

        # Add to total
        total += labels.size(0)

        # Add to correctly classified images
        success += (predicted == labels.data).sum()
    # end for

    # Print and save loss
    print(u"Test Epoch {} Loss : {} Accuracy : {}".format(epoch, float(loss.data), success / total))
    test_losses[epoch] = float(loss.data)
    test_accuracies[epoch] = success / total
# end for

# Show losses
plt.plot(train_losses.numpy(), c='b')
plt.plot(test_losses.numpy(), c='r')
plt.show()

# Show accuracies
plt.plot(train_accuracies.numpy(), c='b')
plt.plot(test_accuracies.numpy(), c='r')
plt.show()
