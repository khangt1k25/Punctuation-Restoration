import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from preprocessing import MyDataset, dataset_batch_iter
from models import RNNModel

dataset = MyDataset(data_path='./train.txt', label_path='./label.txt')
vocab_size = dataset.vocab_size
embedding_size = 16
num_categories = 4
hidden_dim = 16
n_layers = 1
learning_rate = 0.001
batch_size = 2
length = 32
model = RNNModel(vocab_size, embedding_size,
                 num_categories, hidden_dim, n_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
nll_loss = nn.NLLLoss()


def test(model, test_dataset, batch_size=1):
    model.eval()
    test_loss = 0.
    hidden = model.init_hidden(batch_size)

    for batch, data in enumerate(dataset_batch_iter(test_dataset, batch_size)):
        input_tensor = torch.Tensor(data['data']).type(torch.LongTensor)
        target_tensor = torch.Tensor(data['label']).type(torch.LongTensor)

        output, hidden = model(input_tensor, hidden)
        prediction = output.argmax(dim=-1)

        loss = nll_loss(output.view(-1, num_categories),
                        target_tensor.view(-1))

        test_loss += loss.item()

        accuracy = torch.sum(prediction == target_tensor)
        accuracy = accuracy/(batch_size*length)
    return test_loss, accuracy


def train_epoch(model, train_dataset, batch_size):
    # training
    model.train()
    hidden = model.init_hidden(batch_size)
    train_loss = 0.
    for batch, data in enumerate(dataset_batch_iter(train_dataset, batch_size)):
        input_tensor = torch.Tensor(data['data']).type(torch.LongTensor)
        target_tensor = torch.Tensor(data['label']).type(torch.LongTensor)

        optimizer.zero_grad()
        output, hidden = model(input_tensor, hidden)
        hidden = Variable(hidden.data, requires_grad=True)

        loss = nll_loss(output.view(-1, num_categories),
                        target_tensor.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss


def infere(model, input_tensor):
    # infering
    model.eval()
    batch_size = input_tensor.shape[0]
    hidden = model.init_hidden(batch_size)
    output, _ = model(input_tensor, hidden)
    prediction = output.argmax(dim=-1)
    return prediction


def train(epochs, train_dataset, test_dataset):

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataset, batch_size)

        if epoch % 10 == 0:
            test_loss, accuracy = test(model, test_dataset, batch_size)
            print(
                f"Epoch {epoch} --train loss {train_loss} -- test loss {test_loss}-- test acc {accuracy}")
