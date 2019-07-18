rm -rf ./db
rm -f *.h5
rm -f model.eps

python dataset_builder.py

python autoencoder.py

python fer_autoenc.py

python fer_autoenc_tests.py