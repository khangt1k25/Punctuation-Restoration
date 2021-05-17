1. Plan chạy

- Chạy lấy 2 bộ dữ liệu [100k/10k] và [50k/5k]

- Sử dụng bộ [50k/5k] để làm so sánh giữa các mô hình:
cho ra các mỗi phiên bản 2 file train, test score (confix để đánh tên rõ ràng). Các tham số còn lại như sau nlayers = 2, embedding_size = 512, hidden_dim= 256, output_dim=4, chỉ đổi length, chạy trong 100 epochs.

    - RNN vs length=32
    - GRU vs length=32
    - RNN vs length=52
    - GRU vs length=52

    

- Sử dụng bộ [100k/10k] để làm so sánh giữa các mô hình:
  cho ra các mỗi phiên bản 2 file train, test score
    - RNN vs length=32
    - GRU vs length=32

- Trong toàn bộ data m đã chạy qua prepocessing, vẽ biểu đồ xem thử độ dài câu như ntn