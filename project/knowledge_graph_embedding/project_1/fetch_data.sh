#从数据服务器上下载模型文件和训练数据
wget ftp://127.0.0.1:1080/data_path  --ftp-user=user --ftp-password=1234567 -r
mv 127.0.0.1:1080/data_path data_path
rm -r 127.0.0.1