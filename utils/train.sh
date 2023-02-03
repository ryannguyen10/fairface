cd ../
mkdir -p data/
cd data/
mkdir -p facial_image
cd facial_image

wget https://www.dropbox.com/s/wdws3b3fjo190sk/self_generated.tar.gz?dl=0 -O self_generated.tar.gz
tar -zxf self_generated.tar.gz
rm self_generated.tar.gz
