import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from model import *

class TokensReader:
    def __init__(self, labelFile, sourceFile):
        self.labelFile = labelFile
        self.sourceFile = sourceFile
        
        self.label_word2index = {}
        self.label_index2word = {}
        self.label_n_words = 0
        
        self.source_word2index = {}
        self.source_index2word = {}
        self.source_n_words = 0
        
        self.setTokens()

    def setTokens(self):
        lF = open(self.labelFile, "r", encoding="utf-8")
        lines = lF.readlines()
        lF.close()
        
        index = 0
        for line in lines:
            if len(line) > 1:
                self.label_word2index[line[:-1]] = index
                self.label_index2word[index] = line[:-1]
                index += 1
        self.label_n_words = index
            
        
        sF = open(self.sourceFile, "r", encoding="utf-8")
        lines = sF.readlines()
        sF.close()
        
        index = 0
        for line in lines:
            if len(line) > 1:
                self.source_word2index[line[:-1]] = index
                self.source_index2word[index] = line[:-1]
                index += 1
        self.source_n_words = index


tokens = TokensReader("label_tokens.txt", "source_tokens.txt")


INPUT_DIM = tokens.source_n_words
OUTPUT_DIM = tokens.label_n_words
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = AttnDecoderRNN('general', OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('b4_atten_model_94f.pt', map_location=device))

def textToTensor(text, tokens):
    indexes = []
    indexes.append(SOF_token)
    words = []
    for i in text:
        if not checkBlank(i):
            words += i.split()
    i = getRel(words)
    indexes += [tokens.source_word2index.get(word, UNK_token) for word in i.split()]
    indexes.append(EOF_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def predict(model, file, tokens):
    model.eval()
    with torch.no_grad():
        source = file.read().strip().split('\n')
        inputTensor = textToTensor(source, tokens)

        emptyTargetTensor = torch.zeros(1, 1, dtype=torch.long, device = device)
        
        outputs = model(inputTensor, emptyTargetTensor, teacher_forcing_ratio=0)

        outputs_tokens = []
        for output in outputs:
            it = torch.argmax(output[0]).item()
            if it != 0 and it != 1 and it != 2:
                outputs_tokens .append(it)    
            
        b = ' '.join([tokens.label_index2word[idx] for idx in outputs_tokens])
        return b

root = tk.Tk()
root.withdraw()

def pathToFilename(path):
    filename = ''
    for x in path:
        filename += x
        if x == '/':
            filename = ''
    return filename

while(1):
    path = filedialog.askopenfilename()
    try:
        file = open(path, 'r', encoding='utf-8')
    except OSError as e:
        break

    b = predict(model, file, tokens)

##    f = open("testLabel/" + pathToFilename(path), 'r', encoding='utf-8')
##    ref = f.readline()
##    f.close()
    
    file.close()
    
    tk.messagebox.showinfo(title= pathToFilename(path) + "Case Info", message = b)
    #tk.messagebox.showinfo(title= pathToFilename(path) + "Case Info", message = b + "\n" + ref)
