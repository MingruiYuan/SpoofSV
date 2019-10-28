# Spoofing Speaker Verification

This repository replicates the paper [url].

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

Make sure to **overwrite DATA_ROOT_DIR** in **config.json** with your own directory name. **DATA_ROOT_DIR** is the directory where you save the VCTK-Corpus.

```json
"DATA_ROOT_DIR": "/scratch/myuan7/ssv_corpus/VCTK-Corpus/"
```

### 1.4.Dataset Preprocessing

```shell
python -u metagen.py --config_path config.json
```

This script eliminates invalid data and downsample audio files from 48kHz to 22.05kHz. It also generates meta-data of the dataset.

### 1.5.Training

If you would like to train the model from the beginning, please use the following commands. If you would like to use our pre-trained model ([download](https://drive.google.com/open?id=14Yyf5XznRartjxoGZS-e8jP0BR6xZDlP)), please skip this training part. But please note that the pretrained models may not support further training because of the different version of the code. For inference, the pretrained models can work well.

Firstly please modify **config.json**. Make sure to **overwrite SPK_EMB_DIR** and  **SRC_ROOT_DIR**. **SPK_EMB_DIR** is the directory where you save speaker embeddings. **SRC_ROOT_DIR** is the directory of this repository. 

```json
"SPK_EMB_DIR": "/scratch/myuan7/SpoofSV/spk_emb/",
"SRC_ROOT_DIR": "/scratch/myuan7/SpoofSV/"
```

**1.5.1.Train Text2Mel**

In order to discriminate the different training processes we use the start time **ctime** as a tag. The tag **ctime** exists in the name of directories where the checkpoints are saved.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_text2mel -C config.json -T ${ctime} --adversarial --save_spectrogram 
```

**1.5.2.Train SSRN**

You can train Text2Mel and SSRN at the same time.

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} --adversarial --save_spectrogram
```

**1.5.3.Restore from Checkpoints**

In training, checkpoints are saved every **VAL_EVERY_ITER** iterations. You can modify **VAL_EVERY_ITER** in **config.json**. Checkpoints are saved at **./checkpoints/conditional/adversarial/${ctime}/**. If you would like restore training from a checkpoint, please use the following command

```shell
ctime=$(date "+%y-%m-%d_%H-%M-%S")
python -u main.py train_ssrn -C config.json -T ${ctime} -R [/path/to/checkpoint] --adversarial --save_spectrogram 
```

### 1.6.Synthesizing from Test Set

Please firstly modify **INFERENCE_TEXT2MEL_MODEL** and **INFERENCE_SSRN_MODEL** in **config.json**. 

```json
"INFERENCE_TEXT2MEL_MODEL": "./checkpoints/conditional/adversarial/19-08-17_13-05-42/text2mel_iteration_538001.tar.pth",
"INFERENCE_SSRN_MODEL": "./checkpoints/conditional/adversarial/19-08-16_15-21-21/ssrn_iteration_308001.tar.pth"
```

These two models will be used to generate speech. You can use models trained yourself or the pretrained models. Then the following command synthesizes speech from test set. The output audio files are saved at **./samples/${ctime}/**.

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

Please **overwrite ANTISPOOF_DIR** in **config.json**. **ANTISPOOF_DIR** is the directory where you save the ASVspoof2019-LA dataset.

```json
"ANTISPOOF_DIR": "/scratch/myuan7/corpus/ASVspoof2019/LA/"
```

**2.1.2.Generating Data for Spoofing Speaker Verification Systems**

Please also modify **INFERENCE_TEXT2MEL_MODEL** and **INFERENCE_SSRN_MODEL** in **config.json**. These two models will be used to generate speech. Then the following command synthesizes speech from Harvard Sentences. The output audio files are saved at **./test/${ctime}/**. For the text2mel and ssrn models, you can use any checkpoint you have saved or the pretrained models.

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
# Then follow the README in that directory.
```

After running i-vectors and GE2E, you get scores of i-vectors and similarity matrices of GE2E. The following command is used to plot curves of *Spoof Rate (SR)* versus *False Rejection Rate (FRR) in real speech*.

```shell
python -u curve.py --simmat [/path/to/GE2E_simmat] --ivector_score [/path/to/ivector_score]
```

### 2.4.Spoofing Anti-spoofing Systems

```shell
cd anti_spoofing
# Then follow the README in that directory.
```

## Appendix

**./main.py**  main script.

**./metagen.py** data preprocessing and meta-data generation.

**./config.json** configuration file.

**./models/**  structures of Text2Mel, SSRN and discriminators.

**./data/** VCTK datasets.

**./spk_emb/** speaker embeddings.

**./train/** traning pipeline.

**./synthesize.py** synthesizing from test sets.

**./generate_test_utterances.py** synthesizing speech for spoofing experiments from Harvard Sentences.

**./curve.py** plot curves.

**./GE2E/** spoofing Google's GE2E.

**./kaldi_ivector/** spoofing i-vectors