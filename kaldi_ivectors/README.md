# Spoofing I-VECTORS

### 1. Download and Install Kaldi

Please install kaldi according to [http://kaldi-asr.org/doc/](http://kaldi-asr.org/doc/).  It is recommended that creating a new conda environment for kaldi.

Then go to *egs/aishell* in kaldi.  When you are in the directory *aishell*, please copy **v1**. And use the contents in this directory (conf/, local/, run.sh) to replace the corresponding contents in the v1-copy directory.

### 2.Train and Test I-VECTORS

Please modify **src_root** and **ctime** in *run.sh*. **src_root** is the root directory of this SpoofSV repository. **ctime** is determined by the directory where you save the test speech.

```shell
src_root=/scratch/myuan7/program/SpoofSV
ctime=19-09-12_09-08-12
```

Then if it is the first time to work on some speaker split scheme, you need to train the i-vectors system and then perform testing. Please use the following command:

```shell
./run.sh 0
```

If it is not the first time to work on some speaker split scheme (eg. you have trained i-vectors on a 88-20 speaker split scheme and you generate 88-20 speech with another TTS model), you do not need to train the i-vectors again. Please use the following command:

```shell
./run.sh 1
```

 After running the script, the scores of all utterances are saved at ./ivector_scores/.

### 3.Compute Spoof Rate

After running *run.sh*, the threshold of EER is printed on the screen. Then you can use this threshold and scores to compute Spoof Rate (SR).

```shell
python -u ivector_spoofrate.py -S ivector_scores/${ctime}_mix.score --thres [threshold of EER]
```

If you forgot the threshold, you could compute the threshold of EER again with the following command. Also, please modify the **src_root** and **ctime**.

```shell
src_root=/scratch/myuan7/program/SpoofSV
ctime=19-09-12_09-08-12
```

```
./ivector_eer.sh
```

You will get the threshold of EER. But please note that this is applicable only when you do **not** run the script *run.sh* on other generated speech datasets.