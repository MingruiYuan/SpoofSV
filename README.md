# Spoofing Speaker Verification

This repository is in progress. It reproduces the paper [url].

## Section I Multi-speaker Text-to-speech

### 1.1.Clone this Repository

```shell
git clone https://github.com/MingruiYuan/SpoofSV.git && cd SpoofSV
```

### 1.2.Install Packages

Please create a new environment and install necessary packages.

```shell
conda create -n [env name] python=3.6
pip install -r requirements.txt
```

### 1.3.Download VCTK-Corpus

```shell
wget -bc http://datashare.is.ed.ac.uk/download/DS_10283_2651.zip
unzip DS_10283_2651.zip && unzip VCTK-Corpus.zip
```

Make sure to **overwrite DATA_ROOT_DIR in config.json** with your own directory name. **DATA_ROOT_DIR** is the directory where you save the VCTK-Corpus.

### 1.4.Dataset Preprocessing

```shell
python -u metagen.py --config_path config.json
```

This script eliminates invalid data and downsample audio files from 48kHz to 22.05kHz. It also generates meta-data of the dataset.

### 1.5.Training

If you would like to train the model from the beginning, please use the following commands. If you would like to use our pre-trained model, please skip this training part.

Firstly please modify **config.json**. Make sure to **overwrite SPK_EMB_DIR and SRC_ROOT_DIR**. **SPK_EMB_DIR** is the directory where you save speaker embeddings (provided at ./spk_emb). **SRC_ROOT_DIR** is the directory of this repository. 

**1.5.1.Train Text2Mel**

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_text2mel -C config.json -T ${ctime} --adversarial --save_spectrogram 
```

**1.5.2.Train SSRN**

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} --adversarial --save_spectrogram
```

Please note that we use the time when the training starts to identify different training. The time (${ctime}) is used to save checkpoints and generated speech in different directories.

**1.5.3.Restore from Checkpoints**

In training, checkpoints are saved every **VAL_EVERY_ITER** iterations. You can modify **VAL_EVERY_ITER** in config.json. Checkpoints are saved at **./checkpoints/conditional/adversarial/${ctime}/**. If you would like restore training from a checkpoint, please use the following command

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} -R [/path/to/checkpoint] --adversarial --save_spectrogram 
```

### 1.5.4.Synthesizing from Test Set

Please firstly modify **INFERENCE_TEXT2MEL_MODEL** and **INFERENCE_SSRN_MODEL** in config.json. These two models will be used to generate speech. Then the following command synthesizes speech from test set. The output audio files are saved at **./samples/${ctime}/**.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py synthesize -C config.json -T ${ctime} --adversarial --save_spectrogram
```

## Section II Spoofing Speaker Verification Systems and Anti-spoofing Systems

### 2.1.Generating Data from Harvard Sentences for Spoofing Speaker Verification Systems

**2.1.1.Download ASVspoof 2019 Dataset, Logical Access (LA)**

```shell
wget -bc https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
```

Please **overwrite ANTISPOOF_DIR** in config.json. **ANTISPOOF_DIR** is the directory where you save the ASVspoof2019-LA dataset.

**2.1.2.Generating Data**

Please also modify **INFERENCE_TEXT2MEL_MODEL** and **INFERENCE_SSRN_MODEL** in config.json. These two models will be used to generate speech. Then the following command synthesizes speech from Harvard Sentences. The output audio files are saved at **./test/${ctime}/**. For the text2mel and ssrn models, you can use any checkpoint you have saved or the pretrained models.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u generate_test_utterances.py -C config.json -T ${ctime} --train_spk_num [88/60/42]
```

Within each directory **./test/${ctime}/**, **spoof_data** saves all synthetic speech, **ivector_data** saves data for i-vectors' evaluation. **ge2e_data** saves data for Google's GE2E's evaluation. 

### 2.2.Spoofing I-VECTORS

```shell
cd kaldi_ivectors
# Then follow the README in that directory.
```

### 2.3.Spoofing GE2E

```shell
cd GE2E
# Then follow the README in that direcotry.
```

