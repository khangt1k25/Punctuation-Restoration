## in progress ...

## 

1. Tạo 1 colab notebooks, mount drive vào Mydrive

2. Git clone repo này về

3. Cd vào thư mục mới clone
    
4. Tạo folder rỗng dumps và logs trong thư mục vừa có
   
   . Dumps để lưu lại model còn logs để lưu các score phục vụ vẽ biểu đồ
   . Logs sẽ có 2 file khi chạy xong train, test. trong đó mỗi dòng là (loss, acc, precison_1, recall1, precision2, recall2)


5. Chạy lệnh để train, score sẽ tự lưu vào folder
   
```
!python main.py  
```

Note: 
. M config trực tiếp vào chỗ parse default value ấy, chỉ nên thay đổi các path thôi, đoạn length t đang để là 52 vì thấy 52 hợp lí hơn 32.

. Chạy 2 phiên bản với data 50k / 5k: Với 200 epochs sử dụng RNN, GRU (chỉ đổi tên model!!!)
BiLSTM t se update sau

. Mỗi lần chạy xong 1 phiên bản thì nhớ lưu file logs về local rồi xóa file đó đi chạy lại phiên bản mới vì t dùng lệnh append nên nếu không xóa thì hn cứ viết thêm vào

. M tạo 3 tài khoản google, Mở 3 tab rồi đăng nhập 3 tài khoản, mỗi cái train 1 phiên bản.