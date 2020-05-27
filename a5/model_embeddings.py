#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        # constants
        char_embed_size = 50
        dropout_rate = 0.3
        self.word_embed_size = word_embed_size

        # layers
        self.char_embedding = nn.Embedding(len(vocab.char2id), char_embed_size, padding_idx=vocab.char_pad)
        self.cnn = CNN(char_embed_size, word_embed_size)
        self.highway = Highway(input_size=word_embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # sentence_length, batch_size, max_word_length = input.shape

        # lookup character embedding
        x_emb = self.char_embedding(input)      # (sentence_length, batch_size, max_word_length, char_embed_size)
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.shape

        # reshape
        x_reshaped = x_emb.permute(0, 1, 3, 2)  # (sentence_length, batch_size, char_embed_size, max_word_length)

        # convolutional layer
        x_reshaped = x_reshaped.view(-1, char_embed_size, max_word_length)
        x_conv_out = self.cnn(x_reshaped)       # (sentence_length * batch_size, word_embed_size)

        # highway layer and dropout
        x_highway = self.highway(x_conv_out)    # (sentence_length * batch_size, word_embed_size)
        output = self.dropout(x_highway)
        output = output.view(sentence_length, batch_size, -1)  # (sentence_length, batch_size, word_embed_size)

        return output
        
        ### END YOUR CODE

