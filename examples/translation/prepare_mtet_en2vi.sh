mkdir mtet_en_vi
cd mtet_en_vi

# Download train vietnamese
wget https://storage.googleapis.com/vietai_public/best_vi_translation/v2/train.vi

# Download train english
wget https://storage.googleapis.com/vietai_public/best_vi_translation/v2/train.en


split -l 4000000 train.en
rm train.en
mv xaa train.en
mv xab valid.en


split -l 4000000 train.vi
rm train.vi
mv xaa train.vi
mv xab valid.vi
