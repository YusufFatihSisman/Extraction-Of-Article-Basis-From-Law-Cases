from io import open
import string
import re
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

PAD_token = 0
SOF_token = 1
EOF_token = 2
UNK_token = 3

def checkBlank(line):
    for x in line:
        if x.isalpha() or x.isnumeric():
            return False
    return True

def splitToLine(words, start, end):
    if start < 0:
        start = 0
    if end > len(words):
        end = len(words)
    line = ''
    for i in range(start, end):
        line += words[i]
        line += ' '
    return line

def getRel(words):
    d_count = 0
    count = 0
    art = 0
    retLine = ''

    startIndex = 0
    endIndex = 0
    for index in range(1, len(words)):
        if 'ihlal' in words[index]:
            if endIndex == 0:  
                startIndex = index-2
                endIndex = index+5
            elif index-2 < endIndex:
                endIndex = index+5
            elif index-2 >= endIndex:
                retLine += splitToLine(words, startIndex, endIndex)
                retLine += '\n'
                startIndex = index-2
                endIndex = index+5
        elif words[index].isdigit():
            try:
                if int(words[index])<50 and len(words[index])<3:
                    c = True
                    if index + 2 < len(words):
                        if words[index+2].isdigit():
                            if int(words[index+2])>50 or len(words[index+2])>2:
                                c = False
                    if index + 1 < len(words):
                        if words[index+1].isdigit():
                            if int(words[index+1])>50 or len(words[index+1])>2:
                                c = False
                    if c == True:
                        if endIndex == 0:  
                            startIndex = index-2
                            endIndex = index+5
                        elif index-2 < endIndex:
                            endIndex = index+5
                        elif index-2 >= endIndex:
                            retLine += splitToLine(words, startIndex, endIndex)
                            retLine += '\n'
                            startIndex = index-2
                            endIndex = index+5
            except ValueError:
                flag = False
        
    if endIndex != 0:
        retLine += splitToLine(words, startIndex, endIndex)
    if retLine == '':
        return retLine
    else:
        return retLine

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell

class Attn(nn.Module):
    def __init__(self, method, hid_dim):
        super(Attn, self).__init__()
        
        self.method = method
        self.hid_dim = hid_dim

        if self.method == 'general':
            self.linear_in = nn.Linear(self.hid_dim, hid_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs): 
        encoder_outputs = encoder_outputs.transpose(0, 1)
        batch_size, output_len, dimensions = encoder_outputs.size()
        query_len = 1
        
        if self.method == "general":
            encoder_outputs = encoder_outputs.reshape(batch_size * output_len, dimensions)
            encoder_outputs = self.linear_in(encoder_outputs)
            encoder_outputs = encoder_outputs.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(hidden[-1].unsqueeze(1), encoder_outputs.transpose(1, 2).contiguous())

        attention_scores = attention_scores.view(batch_size * query_len, output_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, query_len, output_len)

        context = torch.bmm(attention_weights, encoder_outputs)
        combined = torch.cat((context, hidden[-1].unsqueeze(1)), dim=2)
        return combined, attention_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.attn_model = attn_model
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim*2)
        
        self.rnn = nn.LSTM(emb_dim*2, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if attn_model != 'none':
            self.attn = Attn(attn_model, hid_dim)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output, attn_weights = self.attn(hidden, encoder_outputs)
        output = self.fc_out(output.squeeze(1))
        return output, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device = self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        if self.eval and batch_size == 1:
            t = 1
            outputs = torch.zeros(1, batch_size, trg_vocab_size, device = self.device)
            while(1):
                output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
                top1 = output.argmax(1) 
                output = output.unsqueeze(0)
                outputs = torch.cat((outputs, output), dim=0) 
                if top1 == EOF_token:
                    break
                input = top1
                t += 1
        else: 
            for t in range(1, trg_len):
                output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1) 
                input = trg[t] if teacher_force else top1
        return outputs
