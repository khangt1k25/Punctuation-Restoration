import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import MyDataset, TestDataset, dataset_batch_iter
from models import RNNModel
import numpy as np
import argparse

nll_loss = nn.NLLLoss()


parser = argparse.ArgumentParser()

parser.add_argument('--embedding_size', type=int, default=16, help='embedding dim')
parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dim state of block RNN')
parser.add_argument('--lr', type=float, default=0.0001, help='ADAM learning rate ')
parser.add_argument('--length', type=int, default=32, help='max length of a sentence')
opt = parser.parse_args()

length = opt.length
lr = opt.lr 
embedding_size = opt.embedding_size
hidden_dim = opt.hidden_dim

class Trainer():
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.batch_size = 64
        self.num_categories = 4
        self.save_path = './dumps/model.pt'
        self.length = length

    def train(self, train_dataset, test_dataset, start_epoch, end_epoch):
        # training
        for epoch in range(start_epoch+1, end_epoch+1):
            self.model.train()
            hidden = self.model.init_hidden(self.batch_size)
            train_loss = 0.
            for batch, data in enumerate(dataset_batch_iter(train_dataset, self.batch_size)):
                input_tensor = torch.Tensor(data['data']).type(torch.LongTensor).to(self.device)
                target_tensor = torch.Tensor(data['label']).type(torch.LongTensor).to(self.device) 
                
                output, hidden = self.model(input_tensor, hidden)
                hidden = Variable(hidden.data, requires_grad=True).to(self.device)
                loss = nll_loss(output.view(-1, self.num_categories),
                                target_tensor.view(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= (batch+1)

            if epoch % 10 == 0:
                test_loss, test_accuracy = self.evaluate(test_dataset)
                train_loss, train_accuracy = self.evaluate(train_dataset)
                print(
                    f"Epoch {epoch} -- train loss: {train_loss} -- test loss: {test_loss} -- train acc: {train_accuracy} -- test acc: {test_accuracy}"
                )

        # saving the last
        id2word = train_dataset.id2word
        word2id = train_dataset.word2id
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "id2word": id2word,
            "word2id": word2id
        }, self.save_path)
    
    def evaluate(self, valid_dataset):
        self.model.eval()
        test_loss = 0.
        hidden = self.model.init_hidden(self.batch_size)

        correct = 0
        for batch, data in enumerate(dataset_batch_iter(valid_dataset, self.batch_size)):
            input_tensor = torch.Tensor(data['data']).type(torch.LongTensor).to(self.device)
            target_tensor = torch.Tensor(data['label']).type(torch.LongTensor).to(self.device)

            output, hidden = self.model(input_tensor, hidden)
            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            correct += torch.sum(prediction == target_tensor).item()
        
        
        accuracy = correct/(self.batch_size*self.length*(batch+1))

        return test_loss, accuracy
        

if __name__ == '__main__':
    train_dataset = MyDataset('./demo_data/text.txt', './demo_data/label.txt')

    model = RNNModel(
        vocab_size=train_dataset.vocab_size,
        embedding_size=embedding_size,
        output_size=4, 
        hidden_dim=hidden_dim,
        n_layers=1,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, optimizer, device)

    

    test_dataset = TestDataset('./demo_data/testtext.txt', label_path='./demo_data/testlabel.txt', word2id=train_dataset.word2id, id2word=train_dataset.id2word)

    
    trainer.train(train_dataset, test_dataset, start_epoch=0, end_epoch=500)

