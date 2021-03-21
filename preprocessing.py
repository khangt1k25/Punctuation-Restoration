from torch.utils.data import Dataset
from keras.preprocessing import text
from keras.preprocessing import sequence


class MyDataset(Dataset):
    def __init__(self, data_path, label_path):

        self.data_path = data_path
        self.label_path = label_path

        with open(self.data_path, 'r') as f:
            sents = f.read().splitlines()

        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(sents)
        self.word2id = tokenizer.word_index
        self.word2id['<pad>'] = 0
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)

        print(self.word2id)

        self.sents = [sent.split() for sent in sents]
        self.sents = [[self.word2id[word] for word in sent]
                      for sent in self.sents]

        self.sents = sequence.pad_sequences(
            self.sents, maxlen=32, padding="post")

        with open(self.label_path, 'r') as f:
            labels = f.read().splitlines()

        self.labels = [list(map(int, label.split())) for label in labels]
        self.labels = sequence.pad_sequences(
            self.labels, maxlen=32, padding="post", value=3)

    def __getitem__(self, index):

        return {'data': self.sents[index], 'label': self.labels[index]}

    def __len__(self):
        return len(self.labels)


def dataset_batch_iter(dataset, batch_size):
    b_words = []
    b_labels = []
    for data in dataset:
        b_words.append(data['data'])
        b_labels.append(data['label'])

        if len(b_words) == batch_size:
            yield {'data': np.array(b_words, dtype=int), 'label': np.array(b_labels, dtype=int)}
            b_words, b_labels = [], []
