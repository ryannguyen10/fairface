echo 'Building directory structure...'
cd ../
mkdir -p data/
cd data/
mkdir -p facial_image
cd facial_image

echo 'Directories done. Downloading data...'
wget https://www.dropbox.com/s/ggq15lzghbeqxr3/fairface-img-margin025-trainval.zip?dl=0
mv 5793795 fairface-img-margin025-trainval.zip

echo 'Downloaded, unzipping files...'
unzip fairface-img-margin025-trainval.zip

echo 'moving files to correct directories...'
mv train/* ./ 

echo 'Removing unwanted files...'

rm -r train

echo 'All done!'

