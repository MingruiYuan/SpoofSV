#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import argparse
import librosa
import numpy as np
from hparam import hparam as hp

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))
audio_path.sort(key=lambda x:x[-3:])

def save_spectrogram_tisv(train_spk_num, enroll_num, eval_num):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num = train_spk_num            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%i)
        print(folder)
        utterances_spec = []
        eval_spec = []
        tooshort_list = []
        if i < train_speaker_num:
            utts_list = os.listdir(folder)[:100]
        else:
            utts_list = os.listdir(folder)
            utts_list.sort(key = lambda x:x[:-4])
        for k, utter_name in enumerate(utts_list):
            print('utter name:', utter_name)
            if utter_name[-4:] == '.wav':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio          
                utter, _ = librosa.effects.trim(utter, 30)

                if len(utter) > utter_min_len:
                    S = librosa.core.stft(y=utter, n_fft=hp.data.nfft, win_length=int(hp.data.window*sr), hop_length=int(hp.data.hop*sr))
                    S = np.abs(S)**2
                    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)

                    if i>=train_speaker_num and k>=enroll_num:
                        eval_spec.append(S[:, :hp.data.tisv_frame])
                        eval_spec.append(S[:, -hp.data.tisv_frame:])

                    else:
                        utterances_spec.append(S[:, :hp.data.tisv_frame])
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])

                else:
                    print('Too short!!')
                    tooshort_list.append(k)

        if i>=train_speaker_num:
            enroll_egs_num = len(utterances_spec)
            eval_egs_num = len(eval_spec)
            if 2*enroll_num - enroll_egs_num > 0:
                for p in range(enroll_num-enroll_egs_num//2):
                    loc1 = np.random.randint(0, enroll_egs_num//2)
                    loc2 = np.random.randint(0, enroll_egs_num//2)
                    utterances_spec.append(utterances_spec[loc1])
                    utterances_spec.append(utterances_spec[loc2])

            if 2*eval_num - eval_egs_num > 0:
                for p in range(eval_num-eval_egs_num//2):
                    loc1 = np.random.randint(0, eval_egs_num//2)
                    loc2 = np.random.randint(0, eval_egs_num//2)
                    eval_spec.append(eval_spec[loc1])
                    eval_spec.append(eval_spec[loc2])

            utterances_spec.extend(eval_spec)

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i >= train_speaker_num:
            assert (utterances_spec.shape[0]==2*(enroll_num+eval_num))

        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument('--train_spk_num', type=int, default=88)
    ps.add_argument('--enroll_num', type=int, default=3)
    ps.add_argument('--eval_num', type=int, default=20)
    args = ps.parse_args()
    save_spectrogram_tisv(args.train_spk_num, args.enroll_num+args.eval_num, args.eval_num)