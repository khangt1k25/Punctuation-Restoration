




def infere(model, input_tensor):
    # infering
    model.eval()
    batch_size = input_tensor.shape[0]
    hidden = model.init_hidden(batch_size)
    output, _ = model(input_tensor, hidden)
    prediction = output.argmax(dim=-1)
    return prediction


def restore(tokens, punct):
    id2word = dataset.id2word
    convert = {0: '', 1: ',', 2: '.', 3: ''}
    seq = [id2word[token]+convert[punct[i]] for i, token in enumerate(tokens)]
    seq = ' '.join(seq)
    return seq