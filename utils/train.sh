cd ../
mkdir -p data/
cd data/
mkdir -p facial_image
cd facial_image

wget https://www.dropbox.com/s/g8q9aamjdu4fvt0/train_list_fairface.txt?dl=0 -O train_list_fairface.txt
tar -zxf train_list_fairface.txt
rm train_list_fairface.txt
