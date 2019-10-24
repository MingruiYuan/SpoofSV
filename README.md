# Spoofing Speaker Verification

## Section I Multi-speaker Text-to-speech

#### Clone this Repository

```shell
git clone https://github.com/MingruiYuan/SpoofSV.git && cd SpoofSV
```

#### Install Packages

Please create a new environment and install necessary packages.

```shell
pip install -r requirements.txt
```

#### Download VCTK-Corpus

```shell
wget -bc http://datashare.is.ed.ac.uk/download/DS_10283_2651.zip
unzip -d [your DATA_ROOT_DIR] DS_10283_2651.zip
```

Make sure to overwrite **DATA_ROOT_DIR** in **config.json** with your own directory name.

#### Dataset Preprocessing

```shell
python metagen.py --config_path ./config.json
```

This script will

