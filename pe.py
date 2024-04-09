'''
This is one of the modules you'll submit to the autograder. The positional encoding is implemented in this file.
'''

'''
Note:
Please do not modify any variable name given to you for code completion, especially those that have trainable parameters in torch
'''

import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    We implement the positional encoding as alternating sin and cosine functions.
    In the __init__ function, you simply set the variable pe according the original paper.
    In the forward function, you add the pe to the input
    '''
    def __init__(self, d_model, max_seq_length):
        '''
        Initialize the positional encoding as alternating sin and cosine functions of the dimension position and the model size d_model
        Input:
            d_model (int) - dimension of the Transformer
            
            max_seq_length (int) - maximum sequence length of the Transformer
        '''
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)

        # Iterate through 'rows' of tensor
        for pos in range(max_seq_length):
            # Iterate i = 0,...,d_model / 2
            for i in range(d_model // 2):
                # Set even value
                pe[pos][2 * i] = math.sin(pos * math.exp(2 * i * (-math.log(10000) / d_model)))
                # Set odd value
                pe[pos][2 * i + 1] = math.cos(pos * math.exp(2 * i * (-math.log(10000) / d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Add positional encoding (after appropriate slicing) to the input x

        Input:
            x (torch.Tensor) - Tensor of size B x T x d_model.

        Output:
            torch.Tensor - input with positional encoding added

        """
        # self.pe -> [1, max_seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x