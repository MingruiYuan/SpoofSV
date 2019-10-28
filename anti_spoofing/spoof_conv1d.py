import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
import soundfile as sf
import numpy as np

class ASVspoofDataset(Dataset):

    def __init__(self, cfg, step, time):

        self.cfg = cfg
        # self.own_data = own_data

        suffix = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt' if step == 'train' else 'ASVspoof2019_LA_cm_protocols/customized_data_{}.txt'.format(time)
        proto_fn = cfg['DATA_ROOT_DIR'] + 'data_path/ordinary/wav.path.train'
        with open(proto_fn, 'r') as f:
            audio_fn = f.readlines()
            for k, fn in enumerate(audio_fn):
                audio_fn[k] = fn.strip()
        label_real = torch.ones((20000 if step == 'train' else len(audio_fn)-20000,))
        self.audio_fn = audio_fn[:20000] if step == 'train' else audio_fn[20000:]        

        spoof_fn = cfg['ANTISPOOF_DIR'] + suffix 
        with open(spoof_fn, 'r') as f:
            spoof_data = f.readlines()
        
        spoof_data_num = 0
        mid = 'ASVspoof2019_LA_train' if step == 'train' else time
        for k, proto in enumerate(spoof_data):
            splited_proto = proto.strip().split()
            if splited_proto[-1] == 'spoof':
                self.audio_fn.append(cfg['ANTISPOOF_DIR']+mid+'/flac/'+splited_proto[1]+'.flac')
                spoof_data_num += 1
        label_spoof = torch.zeros((spoof_data_num,))

        self.label = torch.cat((label_real, label_spoof))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.label[idx] == 1:
            audio, sr = librosa.load(self.audio_fn[idx], sr=16000, mono=True)
        else:
            audio, sr = sf.read(self.audio_fn[idx])
        assert sr == 16000

        audio, _ = librosa.effects.trim(audio, 22)
        audio = np.append(audio[0], audio[1:] - self.cfg['PREEMPH']*audio[:-1])
        lin_spec = np.abs(librosa.stft(y=audio, n_fft=self.cfg['STFT']['FFT_LENGTH'], hop_length=self.cfg['STFT']['HOP_LENGTH']))
        mel_filterbank = librosa.filters.mel(sr=sr, n_fft=self.cfg['STFT']['FFT_LENGTH'], n_mels=self.cfg['COARSE_MELSPEC']['FREQ_BINS'])
        mel_spec = np.dot(mel_filterbank, lin_spec)

        maxlin = np.max(lin_spec)
        maxmel = np.max(mel_spec)

        lin_spec_norm = (lin_spec / maxlin)**self.cfg['NORM_POWER']['ANALYSIS']
        mel_spec_norm = (mel_spec / maxmel)**self.cfg['NORM_POWER']['ANALYSIS']

        # Downsampling along time frames.
        reduced_total_time = np.shape(mel_spec)[1] // self.cfg['COARSE_MELSPEC']['REDUCTION']
        reduced_time = [self.cfg['COARSE_MELSPEC']['REDUCTION']*k for k in range(reduced_total_time)]
        reduced_mel_spec = mel_spec_norm[:, reduced_time]
        lin_spec_norm = lin_spec_norm[:, :self.cfg['COARSE_MELSPEC']['REDUCTION']*reduced_total_time]

        return {'data_0':torch.from_numpy(reduced_mel_spec), 'data_1':torch.from_numpy(lin_spec_norm), 'data_2':self.label[idx]}

def collate_pad_3(batch):
    """
    Zero-padding to make the text and spectrogram have the same length.
    Note: As 'P'(denotes padding) pairs with 0 in the vocabulary, we
          automatically use zero padding behind the trial of the text id.
    """
    text_max_length = max([batch[i]['data_1'].size()[1] for i in range(len(batch))])
    spec_max_length = max([batch[i]['data_0'].size()[1] for i in range(len(batch))])

    for i in range(len(batch)):
        len_diff = text_max_length - batch[i]['data_1'].size()[1]
        if len_diff > 0:
            batch[i]['data_1'] = torch.cat((batch[i]['data_1'], torch.zeros((batch[i]['data_1'].size()[0], len_diff))), dim=1)
        len_diff = spec_max_length - batch[i]['data_0'].size()[1]
        if len_diff > 0:
            batch[i]['data_0'] = torch.cat((batch[i]['data_0'], torch.zeros((batch[i]['data_0'].size()[0], len_diff))), dim=1)

    batch_0 = torch.stack([batch[i]['data_0'] for i in range(len(batch))], dim=0)
    batch_1 = torch.stack([batch[i]['data_1'] for i in range(len(batch))], dim=0)
    batch_2 = torch.stack([batch[i]['data_2'] for i in range(len(batch))], dim=0)

    return {'data_0': batch_0, 'data_1': batch_1, 'data_2': batch_2}