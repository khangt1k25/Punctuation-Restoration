import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,
                 hidden_dim, n_layers):

        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.embedding_size = embedding_size


        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)

        self.rnn = nn.RNN(embedding_size, hidden_dim,
                          n_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Sequential( 
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded, hidden)

        output = self.fc(output)
        prob = self.softmax(output)

        return prob, hidden

    def init_hidden(self, batch_size):

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,
                 hidden_dim, n_layers):

        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)

        self.gru = nn.RNN(embedding_size, hidden_dim,
                          n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        embedded = self.embedding(x)

        output, hidden = self.gru(embedded, hidden)

        output = self.fc(output)
        prob = self.softmax(output)

        return prob, hidden

    def init_hidden(self, batch_size):

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,
                 hidden_dim, n_layers, bidirectional=True):

        super(BiLSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)

        self.bilstm = nn.LSTM(embedding_size, hidden_dim,
                              n_layers, batch_first=True, bidirectional=bidirectional, dropout=0.2)

        self.fc = nn.Linear(
            hidden_dim*2 if bidirectional else hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        embedded = self.embedding(x)

        output, hidden = self.bilstm(embedded, hidden)
        
        output = self.fc(output)
        prob = self.softmax(output)

        return prob, hidden

    def init_hidden(self, batch_size):

        hidden = [torch.zeros(
           self.n_layers*2 if self.bidirectional else self.n_layers, batch_size, self.hidden_dim)]*2

        # hidden = torch.zeros(
        #     2, self.n_layers*2 if self.bidirectional else self.n_layers, batch_size, self.hidden_dim)
        return hidden
