cd ../
mkdir -p data/
cd data/
mkdir -p mnist_rotation
cd mnist_rotation

wget https://www.dropbox.com/s/t2k53xzxke4jtq4/train_list_fairface.txt?dl=0 -O train_list_fairface.txt
tar -zxf train_list_fairface.txt
rm train_list_fairface.txt
