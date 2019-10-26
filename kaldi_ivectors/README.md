# Spoofing I-VECTORS

### 1. Download and Install Kaldi

Please install kaldi according to [http://kaldi-asr.org/doc/](http://kaldi-asr.org/doc/).  It is recommended that creating a new conda environment for kaldi.

Then go to *egs/aishell* in kaldi.  When you are in the directory *aishell*, please copy **v1**. And use the contents in this directory (conf/, local/, run.sh) to replace the corresponding contents in the v1-copy directory.

### 2.Modify run.sh

Please modify **src_root** and **ctime** in *run.sh*. **src_root** is the root directory of this SpoofSV repository. **ctime** is determined by the directory where you save the test speech.

```shell
src_root=/scratch/myuan7/program/SpoofSV
ctime=19-09-12_09-08-12
```

