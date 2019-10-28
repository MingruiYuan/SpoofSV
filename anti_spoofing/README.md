# Spoofing Anti-spoofing Systems

*Note: This part has not been verified yet.*

### 1.Training

If you would like to use pretrained models, please skip this part.

```shell
ctime=...
python -u main_spoof_conv1d.py train -T ${ctime} -C ../config.json
```

**${ctime}** should be equal to the time when you generate test data. Checkpoints are saved at **./checkpoints/${ctime}/**.

### 2.Testing

```shell
ctime=... 
python -u main_spoof_conv1d.py dev -T ${ctime} -C ../config.json -R [/path/to/model] --variant [v1/v2]
```

The output scores are saved at **./cm_scores/scores_${ctime}.txt**.