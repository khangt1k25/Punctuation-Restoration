import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import MyDataset, dataset_batch_iter
from models import RNNModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nll_loss = nn.NLLLoss()


class Trainer():
    def __init__(self, data_path="./smalltrain.txt", label_path="./smalllabel.txt",
                 save_folder="./dumps/", num_categories=4, batch_size=64,
                 length=32, learning_rate=0.001, embedding_size=16,
                 hidden_dim=16, n_layers=1, epochs=100):

        self.train_dataset = MyDataset(data_path, label_path)

        self.vocab_size = self.train_dataset.vocab_size
        self.embedding_size = embedding_size
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.length = length
        self.epochs = epochs
        self.save_folder = save_folder

        self.model = RNNModel(self.vocab_size, self.embedding_size,
                              self.num_categories, self.hidden_dim,
                              self.n_layers
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )


    def valid(self, valid_dataset):
        self.model.eval()
        test_loss = 0.
        hidden = self.model.init_hidden(self.batch_size)

        for batch, data in enumerate(dataset_batch_iter(valid_dataset, self.batch_size)):
            input_tensor = torch.Tensor(data['data']).type(torch.LongTensor)
            target_tensor = torch.Tensor(data['label']).type(torch.LongTensor)

            output, hidden = self.model(input_tensor, hidden)
            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            accuracy = torch.sum(prediction == target_tensor)
            accuracy = accuracy/(self.batch_size*self.length)
        return test_loss, accuracy

    def train_epoch(self):
        # training
        self.model.train()
        hidden = self.model.init_hidden(self.batch_size)
        train_loss = 0.
        for batch, data in enumerate(dataset_batch_iter(self.train_dataset, self.batch_size)):
            input_tensor = torch.Tensor(data['data']).type(torch.LongTensor)
            target_tensor = torch.Tensor(data['label']).type(torch.LongTensor)

            self.optimizer.zero_grad()
            output, hidden = self.model(input_tensor, hidden)
            hidden = Variable(hidden.data, requires_grad=True)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss

    def train(self, test_dataset):

        for epoch in range(1, self.epochs+1):
            train_loss = self.train_epoch()

            if epoch % 10 == 0:
                test_loss, test_accuracy = self.valid(test_dataset)
                train_loss, train_accuracy = self.valid(self.train_dataset)
                print(
                    f"Epoch {epoch} -- train loss: {train_loss} -- test loss: {test_loss} -- train acc: {train_accuracy} -- test acc: {test_accuracy}")

        self.saving()

    def saving(self):
        PATH = self.save_folder + "model.pt"
        id2word = self.train_dataset.id2word
        word2id = self.train_dataset.word2id
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "id2words": id2word,
            "word2id": word2id
        }, PATH)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(trainer.train_dataset)
