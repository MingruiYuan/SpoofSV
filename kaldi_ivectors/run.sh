#!/bin/bash
# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Chao Li
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
# Apache 2.0

# This is a shell script that we demonstrate speech recognition using AIShell-1 data.
# it's recommended that you run the commands one by one by copying and pasting into the shell.
# See README.txt for more info on data required.
# Results (EER) are inline in comments below

firstrun=$1

src_root=/scratch/myuan7/program/ACTTS/
ctime=19-09-12_09-08-12

data=${src_root}/test
# data_url=www.openslr.org/resources/33

. ./cmd.sh
. ./path.sh

set -e # exit on error

#local/download_and_untar.sh $data $data_url data_aishell
#local/download_and_untar.sh $data $data_url resource_aishell

if [ ! -d "ivector_scores" ]; then
  mkdir ivector_scores
fi

if [ -d "data" ]; then
  rm -rf data
fi

if [ -d "mfcc" ]; then
  rm -rf mfcc
fi

#Convert test set to pcm format
if [ $firstrun -gt 0 ]
then
	echo "Not the first time to run this script."
else
	echo "The first time to run this script."
fi  

if [ $firstrun -gt 0 ]
then
	for dir in test
	do
		for spk in $( ls ${data}/${ctime}/ivector_data/wav/${dir} )
		do
			for utt in $( ls ${data}/${ctime}/ivector_data/wav/${dir}/${spk} )
			do
				file=${data}/${ctime}/ivector_data/wav/${dir}/${spk}/${utt}
				mv $file ${file%.*}.WAV
				sox ${file%.*}.WAV -t wav -r 16000 -b 16 $file
				rm -f ${file%.*}.WAV
			done
		done
	done
else
	for dir in train test
	do
		for spk in $( ls ${data}/${ctime}/ivector_data/wav/${dir} )
		do
			for utt in $( ls ${data}/${ctime}/ivector_data/wav/${dir}/${spk} )
			do
				file=${data}/${ctime}/ivector_data/wav/${dir}/${spk}/${utt}
				mv $file ${file%.*}.WAV
				sox ${file%.*}.WAV -t wav -r 16000 -b 16 $file
				rm -f ${file%.*}.WAV
			done
		done
	done
fi

# Data Preparation
local/aishell_data_prep.sh $data/${ctime}/ivector_data/wav $data/${ctime}/ivector_data/transcript

# Now make MFCC  features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

if [ $firstrun -gt 0 ]
then
	for x in test; do
	  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/$x exp/make_mfcc/$x $mfccdir
	  sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
	  utils/fix_data_dir.sh data/$x
	done
else
	for x in train test; do
	  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/$x exp/make_mfcc/$x $mfccdir
	  sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
	  utils/fix_data_dir.sh data/$x
	done
fi

if [ $firstrun -eq 0 ]
then
	#train diag ubm
	sid/train_diag_ubm.sh --nj 2 --cmd "$train_cmd" --num-threads 8 \
	 data/train 1024 exp/diag_ubm_1024

	#train full ubm
	sid/train_full_ubm.sh --nj 2 --cmd "$train_cmd" data/train \
	 exp/diag_ubm_1024 exp/full_ubm_1024

	#train ivector
	sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 10G" \
	 --num-iters 5 exp/full_ubm_1024/final.ubm data/train \
	 exp/extractor_1024

	#extract ivector
	sid/extract_ivectors.sh --cmd "$train_cmd" --nj 2 \
	 exp/extractor_1024 data/train exp/ivector_train_1024

	#train plda
	$train_cmd exp/ivector_train_1024/log/plda.log \
	 ivector-compute-plda ark:data/train/spk2utt \
	 'ark:ivector-normalize-length scp:exp/ivector_train_1024/ivector.scp  ark:- |' \
	 exp/ivector_train_1024/plda
fi

#split the test to enroll and eval
mkdir -p data/test/enroll data/test/eval
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/enroll
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/eval
local/split_data_enroll_eval.py data/test/utt2spk  data/test/enroll/utt2spk  data/test/eval/utt2spk
trials=data/test/aishell_speaker_ver.lst
local/produce_trials.py data/test/eval/utt2spk $trials
utils/fix_data_dir.sh data/test/enroll
utils/fix_data_dir.sh data/test/eval

#extract enroll ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 2 \
  exp/extractor_1024 data/test/enroll  exp/ivector_enroll_1024
#extract eval ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 2 \
  exp/extractor_1024 data/test/eval  exp/ivector_eval_1024

#compute plda score
$train_cmd exp/ivector_eval_1024/log/plda_score.log \
  ivector-plda-scoring --num-utts=ark:exp/ivector_enroll_1024/num_utts.ark \
  exp/ivector_train_1024/plda \
  ark:exp/ivector_enroll_1024/spk_ivector.ark \
  "ark:ivector-normalize-length scp:exp/ivector_eval_1024/ivector.scp ark:- |" \
  "cat '$trials' | awk '{print \\\$2, \\\$1}' |" exp/trials_out

#compute eer
echo "******EER and theshold on mixed speech******"
awk '{print $3}' exp/trials_out | paste - $trials | awk '{print $1, $4}' | compute-eer -
mv exp/trials_out ivector_scores/${ctime}_mix.score

mv ${data}/${ctime}/ivector_data/wav/test ${data}/${ctime}/ivector_data/test_mix
mv ${data}/${ctime}/ivector_data/test_nospoof ${data}/${ctime}/ivector_data/wav/test
mv $data/${ctime}/ivector_data/transcript/VCTK-transcript.txt $data/${ctime}/ivector_data/trans_mix.txt
mv $data/${ctime}/ivector_data/VCTK-transcript_nospoof.txt $data/${ctime}/ivector_data/transcript/VCTK-transcript.txt

rm -rf data mfcc

for dir in test
do
	for spk in $( ls ${data}/${ctime}/ivector_data/wav/${dir} )
	do
		for utt in $( ls ${data}/${ctime}/ivector_data/wav/${dir}/${spk} )
		do
			file=${data}/${ctime}/ivector_data/wav/${dir}/${spk}/${utt}
			mv $file ${file%.*}.WAV
			sox ${file%.*}.WAV -t wav -r 16000 -b 16 $file
			rm -f ${file%.*}.WAV
		done
	done
done

local/aishell_data_prep.sh $data/${ctime}/ivector_data/wav $data/${ctime}/ivector_data/transcript

for x in test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/$x exp/make_mfcc/$x $mfccdir
  sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x
done

#split the test to enroll and eval
mkdir -p data/test/enroll data/test/eval
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/enroll
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/eval
local/split_data_enroll_eval.py data/test/utt2spk  data/test/enroll/utt2spk  data/test/eval/utt2spk
trials=data/test/aishell_speaker_ver.lst
local/produce_trials.py data/test/eval/utt2spk $trials
utils/fix_data_dir.sh data/test/enroll
utils/fix_data_dir.sh data/test/eval

#extract enroll ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 2 \
  exp/extractor_1024 data/test/enroll  exp/ivector_enroll_1024
#extract eval ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 2 \
  exp/extractor_1024 data/test/eval  exp/ivector_eval_1024

#compute plda score
$train_cmd exp/ivector_eval_1024/log/plda_score.log \
  ivector-plda-scoring --num-utts=ark:exp/ivector_enroll_1024/num_utts.ark \
  exp/ivector_train_1024/plda \
  ark:exp/ivector_enroll_1024/spk_ivector.ark \
  "ark:ivector-normalize-length scp:exp/ivector_eval_1024/ivector.scp ark:- |" \
  "cat '$trials' | awk '{print \\\$2, \\\$1}' |" exp/trials_out

#compute eer
echo "******EER and threshold on real speech******"
awk '{print $3}' exp/trials_out | paste - $trials | awk '{print $1, $4}' | compute-eer -
mv exp/trials_out ivector_scores/${ctime}_nospoof.score

mv ${data}/${ctime}/ivector_data/wav/test ${data}/${ctime}/ivector_data/test_nospoof
mv ${data}/${ctime}/ivector_data/test_mix ${data}/${ctime}/ivector_data/wav/test
mv $data/${ctime}/ivector_data/transcript/VCTK-transcript.txt $data/${ctime}/ivector_data/VCTK-transcript_nospoof.txt
mv $data/${ctime}/ivector_data/trans_mix.txt $data/${ctime}/ivector_data/transcript/VCTK-transcript.txt

#rm -rf data mfcc

exit 0