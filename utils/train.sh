cd ../
mkdir -p data/
cd data/
mkdir -p mnist_rotation
cd mnist_rotation

wget https://www.dropbox.com/s/t2k53xzxke4jtq4/fairface_label_train.txt?dl=0 -O fairface_label_train.txt
tar -zxf fairface_label_train.txt
rm fairface_label_train.txt
