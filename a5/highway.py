#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1f

    def __init__(self, input_size: int):
        """ Init Highway layer.

        @param input_size (int): Input size (dimensionality)
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Takes a minibatch of input and compute the output of running the input through a highway layer.

        @param x (Tensor): Tensor of a minibatch of inputs of shape (batch_size, input_size)
        @returns x_highway (Tensor): Tensor of the result of running the input through the highway layer of
                                     the same shape as input
        """
        x_proj = F.relu(self.projection(x))
        x_gate = torch.sigmoid(self.gate(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        return x_highway

    # END YOUR CODE
