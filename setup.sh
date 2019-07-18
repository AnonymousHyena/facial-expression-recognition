virtualenv -p python3 fer_env
source fer_env/bin/activate
pip install -r requirements.txt

python dataset_builder.py

python autoencoder.py

python fer_autoenc.py

python fer_autoenc_tests.py