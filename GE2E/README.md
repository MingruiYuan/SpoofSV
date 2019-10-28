# Spoofing GE2E

This part refers to [https://github.com/HarryVolek/PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification).

Please first modify **./config/config.yaml**. Overwrite **unprocessed_data** with your own directory. It should be the directory where you save the generated data for GE2E.

```yaml
unprocessed_data: '/scratch/myuan7/program/ACTTS/test/19-09-12_09-11-19/ge2e_data/*/*.wav'
```

### 1.Data Preprocessing

Please firstly modify **train_path** and **test_path** in config.yaml according to your own code. After preprocessing the features are saved at these directories.

```yaml
data:
    train_path: './train_tisv_19-09-12_09-11-19'
    test_path: './test_tisv_19-09-12_09-11-19'
```

Then you need to specify the number of train speakers which should be equal to the generated data.

```shell
python -u data_preprocess.py --train_spk_num [88/60/42]
```

### 2.Training

If you would like to use the pretrained models [(download)](https://drive.google.com/open?id=1DOjkQ63Pq0x399Jkmy-KIDUcRUQHJiJK), please skip this part.

Before training, firstly set the config.yaml to training mode and choose proper devices.

```yaml
training: !!bool "true"
device: "cuda"
```

As we have preprocessed data, please use preprocessed data. Also, please do not restore a checkpoint.

```yaml
data:
    data_preprocessed: !!bool "true" 
train:
    restore: !!bool "false"
```

And remember to specify a directory to save checkpoints and log file.

```yaml
train:
    log_file: './speech_id_checkpoint_88/Stats'
    checkpoint_dir: './speech_id_checkpoint_88'
```

Run the following command to start training:

```shell
python -u train_speech_embedder.py
```

### 3.Testing

Before testing, firstly set the config.yaml to testing mode and choose proper devices.

```yaml
training: !!bool "false"
device: "cuda"
```

As we have preprocessed data, please use preprocessed data. And you need to specify a model which is used in testing. This model can be a model trained yourself or a pretrained model.

```yaml
data:
    data_preprocessed: !!bool "true" 
model: 
    model_path: './speech_id_checkpoint_88/final_epoch_950_batch_id_10.model'   
```

And remember to specify a place to save simmats (similarity matrices) which are used to compute EER threshold and spoof rate.

```yaml
save_simmat_dir: './simmat_19-09-12_09-11-19'
```

Make sure the number of test speakers equal to the generated data.

```yaml
test:
    N : 20  # This should be 20 (S1), 48(S2) or 66(S3)
```

Run the following command to perform tests:

```shell
python -u train_speech_embedder.py
```

EER and Spoof Rate (SR) will be printed on the screen.