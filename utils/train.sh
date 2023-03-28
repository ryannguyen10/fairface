cd ../
mkdir -p data/
cd data/
mkdir -p facial_image
cd facial_image

wget https://www.dropbox.com/s/rlvmejzu7qh1e89/ezyZip.tar.gz?dl=0 -O dataset.tar.gz
tar -zxf dataset.tar.gz
rm dataset.tar.gz
