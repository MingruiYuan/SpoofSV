# Spoofing Speaker Verification

## Section I Multi-speaker Text-to-speech

### Clone this Repository

```shell
git clone https://github.com/MingruiYuan/SpoofSV.git && cd SpoofSV
```

### Install Packages

Please create a new environment and install necessary packages.

```shell
conda create -n [env name] python=3.6
pip install -r requirements.txt
```

### Download VCTK-Corpus

```shell
wget -bc http://datashare.is.ed.ac.uk/download/DS_10283_2651.zip
unzip DS_10283_2651.zip && unzip VCTK-Corpus.zip
```

Make sure to **overwrite *DATA_ROOT_DIR* in config.json** with your own directory name.

### Dataset Preprocessing

```shell
python -u metagen.py --config_path ./config.json
```

This script eliminates invalid data and downsample audio files from 48kHz to 22.05kHz. It also generates meta-data of the dataset.

### Training

If you would like to train the model from the beginning, please use the following script. If you would like to use our pre-trained model, please skip this part.

**Train Text2Mel**

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_text2mel -C config.json -T ${ctime} --adversarial --save_spectrogram 
```

**Train SSRN**

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} --adversarial --save_spectrogram
```

