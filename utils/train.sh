cd ../
mkdir -p data/
cd data/
mkdir -p facial_image
cd facial_image

wget https://www.dropbox.com/s/gafl7acfi1xlqni/fairface-img-margin025-trainval.zip?dl=0 -O fairface-img-margin025-trainval.zip
unzip fairface-img-margin025-trainval.zip
mv train fairface_train_data

rm fairface-img-margin025-trainval.zip
rm fairface_train_data
