
# load cái model đã lưu, trong đó t có lưu kèm word2id và id2word

# vocab_size = len(word2id) 

# words = torch.arange(0, vocab_size).view(-1, 1) // tạo 1 cái tensor chỉ số ntn để feed qua embedding layer

# embedding_layer = model.embedding
# embedded = embedding_layer(words) // sẽ được vector embedded là word embedding sau khi train của bộ vocab  


# Sau đó sử dụng embedded vào PCA để plot các từ lên 2D xem sao, dễ thôi chuyển shape, về numpy, rồi gọi thư viện



Note:
# shape: words [vocabsize, 1]
#        embedded [vocabsize, embedding_dim]s


# Kết quả mong muốn:
# như hình
# https://raw.githubusercontent.com/sonvx/word2vecVN/master/images/w2vecVN_tb.png