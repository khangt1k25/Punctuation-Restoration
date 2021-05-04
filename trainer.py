import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets import dataset_batch_iter

nll_loss = nn.NLLLoss()



class Trainer():
    def __init__(self, model, optimizer, device, save_model_path, name_log_dir):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.num_categories = self.model.output_size
        self.save_model_path = save_model_path
        self.writer = SummaryWriter(log_dir=name_log_dir)

    def train(self, train_dataset, test_dataset, batch_size, start_epoch, end_epoch):

        # training
        for epoch in range(start_epoch, end_epoch+1):
            self.model.train()
            hidden = self.model.init_hidden(batch_size)
            train_loss = 0.
            num_batch = 0
            for batch, data in enumerate(dataset_batch_iter(train_dataset, batch_size)):
                
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
                num_batch = batch+1
            
            
            test_matrix, test_loss, test_accuracy = self.cal_score(test_dataset)
            train_matrix, train_loss, train_accuracy = self.cal_score(train_dataset)
            
            print(test_matrix)
            print(train_matrix)

            if epoch % 10 == 0:
                ## shuffle evaluate ??
                test_loss, test_accuracy = self.evaluate(test_dataset)
                train_loss, train_accuracy = self.evaluate(train_dataset)
                

                self.writer.add_scalars('Loss',{"train_loss": train_loss, "test_loss": test_loss}, epoch)
                self.writer.add_scalars("Accuracy", {"train _acc": train_accuracy, "test_acc":test_accuracy}, epoch)

                print(f"\nEpoch {epoch} -- train loss: {train_loss} -- test loss: {test_loss} -- train acc: {train_accuracy} -- test acc: {test_accuracy}\n")

                # saving the last
                self.saving(train_dataset, epoch)
                
            else:
                train_loss /= num_batch
                print(f"\nEpoch {epoch} -- train loss: {train_loss}\n")
    
    def saving(self, train_dataset, epoch):
        filename = self.save_model_path+str(epoch)+'.pt'
        id2word = train_dataset.id2word
        word2id = train_dataset.word2id
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "id2word": id2word,
            "word2id": word2id
        }, filename)
        print(f'\nSaving model successfully at epoch {epoch}\n')
    
    def evaluate(self, valid_dataset):
        batch_size = 64
        self.model.eval()
        test_loss = 0.
        hidden = self.model.init_hidden(batch_size)

        correct = 0
      
        for batch, data in enumerate(dataset_batch_iter(valid_dataset, batch_size)):
    
            input_tensor = torch.Tensor(data['data']).type(torch.LongTensor).to(self.device)
            target_tensor = torch.Tensor(data['label']).type(torch.LongTensor).to(self.device)

            output, hidden = self.model(input_tensor, hidden)
            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            correct += torch.sum(prediction == target_tensor).item()


        length = valid_dataset.length

        #print("batch=", batch)
        accuracy = correct/(batch_size*length*(batch+1))

        return test_loss, accuracy
        
    def load(self, epoch):
        try:
            filename = self.save_model_path+str(epoch)+'.pt'
            
            checkpoint = torch.load(filename)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f'\n Loading model successfully at epoch {epoch}\n')
        except:
            print(f'\n Loading model fail at epoch {epoch}\n')
    
    
    def cal_score(self, valid_dataset):
        self.model.eval()
        test_loss = 0.
        batch_size = 64
        hidden = self.model.init_hidden(batch_size)

        cnf_matrix = np.zeros((4, 4))
        for batch, data in enumerate(dataset_batch_iter(valid_dataset, batch_size)):
            input_tensor = torch.tensor(data['data']).type(torch.LongTensor).to(self.device)
            target_tensor = torch.tensor(data['label']).type(torch.LongTensor).to(self.device)

            output, hidden = self.model(input_tensor, hidden)
            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            prediction = prediction.cpu().numpy()
            target_numpy = target_tensor.cpu().numpy()

            for i in range (prediction.shape[0]):
                cnf_matrix[target_numpy[i], prediction[i]] += 1
            
        accuracy = np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()

        return cnf_matrix, test_loss, accuracy