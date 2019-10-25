# Spoofing Speaker Verification

This repository is in progress. It reproduces the paper [url].

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

Make sure to **overwrite DATA_ROOT_DIR in config.json** with your own directory name. **DATA_ROOT_DIR** is the directory where you save the VCTK-Corpus.

### Dataset Preprocessing

```shell
python -u metagen.py --config_path config.json
```

This script eliminates invalid data and downsample audio files from 48kHz to 22.05kHz. It also generates meta-data of the dataset.

### Training

If you would like to train the model from the beginning, please use the following commands. If you would like to use our pre-trained model, please skip this training part.

Firstly please modify **config.json**. Make sure to **overwrite SPK_EMB_DIR and SRC_ROOT_DIR**. **SPK_EMB_DIR** is the directory where you save speaker embeddings (provided at ./spk_emb). **SRC_ROOT_DIR** is the directory of this repository. 

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

Please note that we use the time when the training starts to identify different training. The time (${ctime}) is used to save checkpoints and generated speech in different directories.

**Restore from Checkpoints**

In training, checkpoints are saved every **VAL_EVERY_ITER** iterations. You can modify **VAL_EVERY_ITER** in config.json. Checkpoints are saved at **./checkpoints/conditional/adversarial/${ctime}/**. If you would like restore training from a checkpoint, please use the following command

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} -R [/path/to/checkpoint] --adversarial --save_spectrogram 
```

### Synthesizing from Test Set

Please firstly modify **INFERENCE_TEXT2MEL_MODEL** and **INFERENCE_SSRN_MODEL** in config.json. Then the following command synthesizes from test set. The output audio files are saved at **./samples/${ctime}**.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py synthesize -C config.json -T ${ctime} --adversarial --save_spectrogram
```

### Generating Data from Harvard Sentences for Spoofing Speaker Verification Systems

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u generate_test_utterances.py -C config.json -T ${ctime} --train_spk_num [88/60/42]
```

