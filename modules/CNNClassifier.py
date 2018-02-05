#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# CNN classifier
class CNNClassifier(nn.Module):
    """
    CNN classifier
    """

    # Constructor
    def __init__(self, dropout=False):
        """
        Constructor
        """
        super(CNNClassifier, self).__init__()

        # Properties
        self.dropout = dropout

        # First 1-D convolution layer
        self.conv1_lexical = nn.Conv1d(1, 10, 10)
        self.conv1_topic = nn.Conv1d(1, 10, 10)
        self.conv1_bigrams = nn.Conv1d(1, 10, 10)
        self.conv1_semantic = nn.Conv1d(1, 10, 10)

        # Max-pool layer
        self.pool_lexical = nn.MaxPool1d(4, 2)
        self.pool_topic = nn.MaxPool1d(4, 2)
        self.pool_bigrams = nn.MaxPool1d(4, 2)
        self.pool_semantic = nn.MaxPool1d(4, 2)

        # Second 1-D convolution layer
        self.conv2_lexical = nn.Conv1d(10, 5, 10)
        self.conv2_topic = nn.Conv1d(10, 5, 4)
        self.conv2_bigrams = nn.Conv1d(10, 5, 10)
        self.conv2_semantic = nn.Conv1d(10, 5, 10)

        # First linear layer
        self.linear_layer1 = nn.Linear(5 * 12 + 5 * 4 + 5 * 3 + 5 * 135, 100)

        # Second linear layer
        self.linear_layer2 = nn.Linear(100, 2)
    # end __init__

    # Forward pass
    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        # Features
        lexical_features = torch.cat((x[:, :, 0:31], x[:, :, 93:107], x[:, :, 407:416]), dim=2)
        topic_features = x[:, :, 31:57]
        bigram_features = x[:, :, 57:93]
        semantic_features = x[:, :, 107:407]

        # Firt 1-D convolution layer
        xl = F.relu(self.conv1_lexical(lexical_features))
        xt = F.relu(self.conv1_topic(topic_features))
        xb = F.relu(self.conv1_bigrams(bigram_features))
        xs = F.relu(self.conv1_semantic(semantic_features))

        # Max pool layer
        xl = self.pool_lexical(xl)
        xt = self.pool_topic(xt)
        xb = self.pool_bigrams(xb)
        xs = self.pool_semantic(xs)

        # Second 1-D convo layer
        xl = F.relu(self.conv2_lexical(xl))
        xt = F.relu(self.conv2_topic(xt))
        xb = F.relu(self.conv2_bigrams(xb))
        xs = F.relu(self.conv2_semantic(xs))

        # Flatten
        xl = xl.view(-1, 5 * 12)
        xt = xt.view(-1, 5 * 4)
        xb = xb.view(-1, 5 * 3)
        xs = xs.view(-1, 5 * 135)

        # If dropout
        if self.dropout:
            xl = F.dropout(xl, self.training)
            xt = F.dropout(xt, self.training)
            xb = F.dropout(xb, self.training)
            xs = F.dropout(xs, self.training)
        # end if

        # Joined
        xj = torch.cat((xl, xt, xb, xs), dim=1)

        # First linear layer
        xj = F.relu(self.linear_layer1(xj))

        # Second linear layer
        xj = F.relu(self.linear_layer2(xj))

        return xj
    # end forward

# end CNNClassifier
