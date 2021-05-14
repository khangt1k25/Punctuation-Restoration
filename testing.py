from datasets import TestDataset, MyDataset, dataset_batch_iter
import torch
import torch.nn as nn
from models import RNNModel
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

train_dataset = MyDataset("./demo_data/demo_text.txt",
                          "./demo_data/demo_label.txt", 32)
model = RNNModel(vocab_size=train_dataset.vocab_size, embedding_size=256, output_size=4,
                 hidden_dim=512, n_layers=1)

hidden = model.init_hidden(64)
correct = 0.
cnf_matrix = np.zeros((4, 4), dtype=int)
b= 0

for batch, data in enumerate(dataset_batch_iter(train_dataset, 64)):
    
    input_tensor = torch.Tensor(data['data']).type(
        torch.LongTensor)
    target_tensor = torch.Tensor(data['label']).type(
        torch.LongTensor)

    output, hidden = model(input_tensor, hidden)
    prediction = output.argmax(dim=-1)


    correct += (torch.sum(prediction == target_tensor).item())
    b += target_tensor.shape[0]

    for t, p in zip(target_tensor.view(-1), prediction.view(-1)):
        cnf_matrix[t.cpu().long(), p.cpu().long()] += 1 


print(cnf_matrix)
