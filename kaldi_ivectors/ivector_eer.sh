#!/bin/bash

# scorefile=$1

. ./cmd.sh
. ./path.sh

src_root=/scratch/myuan7/program/ACTTS
ctime=19-09-12_09-08-12
data=${src_root}/test

# local/aishell_data_prep.sh $data/${ctime}/ivector_data/wav $data/${ctime}/ivector_data/transcript

# for x in test; do
#   steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/$x exp/make_mfcc/$x $mfccdir
#   sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
#   utils/fix_data_dir.sh data/$x
# done

# #split the test to enroll and eval
# mkdir -p data/test/enroll data/test/eval
# cp data/test/{spk2utt,feats.scp,vad.scp} data/test/enroll
# cp data/test/{spk2utt,feats.scp,vad.scp} data/test/eval
# local/split_data_enroll_eval.py data/test/utt2spk  data/test/enroll/utt2spk  data/test/eval/utt2spk
# trials=data/test/aishell_speaker_ver.lst
# local/produce_trials.py data/test/eval/utt2spk $trials
# utils/fix_data_dir.sh data/test/enroll
# utils/fix_data_dir.sh data/test/eval

awk '{print $3}' ivector_scores/${ctime}_nospoof.score | paste - data/test/aishell_speaker_ver.lst | awk '{print $1, $4}' | compute-eer -
