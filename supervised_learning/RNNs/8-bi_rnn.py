#!/usr/bin/env python3
'''
    Script that defines a function def bi_rnn(bi_cell, X, h_0, h_t):
    that performs forward propagation for a bidirectional RNN:
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    '''
        Function that performs forward propagation for a bidirectional RNN

        parameters:
            bi_cell: an instance of BidirectionalCell
            X: data
            h_0: initial hidden state
            h_T: terminal hidden state

        return:
            H: all hidden states
            Y: all outputs
    '''

    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, 2, m, h))
    H[0, 0] = h_0
    H[0, 1] = h_T
    for step in range(t):
        h_prev, y = bi_cell.forward(H[step, 0], X[step])
        H[step + 1, 0] = h_prev
        h_next, y = bi_cell.forward(H[step, 1], y)
        H[step + 1, 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, 2, m, output_shape)
    return (H, Y)
