import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import MyDataset, dataset_batch_iter
from models import RNNModel
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading saved model
MODEL_PATH = './dumps/model.pt'
DATA_PATH = './data/small_traintext.txt'
LABEL_PATH = './data/small_trainlabel.txt'
checkpoint = torch.load(MODEL_PATH)
train_dataset = MyDataset(DATA_PATH, LABEL_PATH)

length = 32
embedding_size = 16
hidden_dim = 16
n_layers = 1
num_categories = 4

model = RNNModel(train_dataset.vocab_size, embedding_size,
                 num_categories, hidden_dim, n_layers).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
id2word = checkpoint['id2words']
word2id = checkpoint['word2id']


def infere(model, input_tensor):
    # infering
    model.eval()
    batch_size = input_tensor.shape[0]
    hidden = model.init_hidden(batch_size)
    output, _ = model(input_tensor, hidden)
    prediction = output.argmax(dim=-1)
    return prediction


def restore(id2word, tokens, punct):
    convert = {0: '', 1: ',', 2: '.', 3: ''}
    seq = [id2word[token]+convert[punct[i]] for i, token in enumerate(tokens)]
    seq = ' '.join(seq)
    return seq


if __name__ == "__main__":
    test_dataset = MyDataset('./data/small_validtext.txt',
                             './data/small_validlabel.txt')
    for batch, data in enumerate(dataset_batch_iter(test_dataset, 16)):
        input_tensor = torch.Tensor(data['data']).type(torch.LongTensor)
        prediction = infere(model, input_tensor)
        for i, punct in enumerate(prediction):
            myseq = restore(id2word, np.array(
                input_tensor[i]), np.array(punct))
            print(myseq)
        break
