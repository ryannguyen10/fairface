cd ../
mkdir -p data/
cd data/
mkdir -p mnist_rotation
cd mnist_rotation

wget https://www.dropbox.com/s/mdqkq3lps4uw59g/train_list_fairface.txt?dl=0 -O train_list_fairface.txt
tar -zxf train_list_fairface.txt
rm train_list_fairface.txt
