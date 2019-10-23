import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
import numpy as np
import json

class dataset(Dataset):

	def __init__(self, cfg, mode='train', pattern='conditional', step='train_text2mel', stage=None, spec_dir=None, NAME='VCTK-Corpus'):
		"""
		cfg:                   configuration file.
		mode:                 'train', 'validate', 'synthesize'.
		pattern:              'universal', 'conditional', or 'ubm-finetune'.
		step:                 'train_text2mel', 'train_ssrn' or 'synthesize'.
		stage:                'ubm' , 'finetune', ignored when pattern is not 'ubm-finetune'.
		"""
		self.cfg = cfg
		self.mode = mode
		self.step = step
		self.spec_dir = spec_dir
		self.name = NAME

		self.root_dir = cfg['DATA_ROOT_DIR']
		self.spkemb_dir = cfg['SPK_EMB_DIR']
		self.vocabulary = cfg['VOCABULARY']
		self.freq_bins = cfg['COARSE_MELSPEC']['FREQ_BINS']
		self.tres_dsmp = cfg['COARSE_MELSPEC']['REDUCTION']
		self.preemph = cfg['PREEMPH']
		self.n_fft = cfg['STFT']['FFT_LENGTH']
		self.hop = cfg['STFT']['HOP_LENGTH']
		self.norm_power = cfg['NORM_POWER']['ANALYSIS']

		self.char2idx = {char: idx for idx, char in enumerate(self.vocabulary)}
		self.char2idx['"'] = len(self.vocabulary)-2

		# list of string, path for all wav files.
		if pattern in ['universal', 'conditional']:
			with open(self.root_dir+'data_path/ordinary/wav.path.'+mode, 'r') as f:
				self.wavlist = f.readlines()
				for k, line in enumerate(self.wavlist):
					line = line.strip('\n')
					self.wavlist[k] = line

			# list of string, path for all txt files.
			with open(self.root_dir+'data_path/ordinary/txt.path.'+mode, 'r') as g:
				self.txtlist = g.readlines()
				for k, line in enumerate(self.txtlist):
					line = line.strip('\n')
					self.txtlist[k] = line

		if pattern == 'ubm-finetune':
			if stage == 'ubm':
				with open(self.root_dir+'data_path/ubm-finetune/wav.path.ubm.'+mode, 'r') as f:
					self.wavlist = f.readlines()
					for k, line in enumerate(self.wavlist):
						line = line.strip('\n')
						self.wavlist[k] = line
				with open(self.root_dir+'data_path/ubm-finetune/txt.path.ubm.'+mode, 'r') as g:
					self.txtlist = g.readlines()
					for k, line in enumerate(self.txtlist):
						line = line.strip('\n')
						self.txtlist[k] = line

			if stage == 'finetune':
				with open(self.root_dir+'data_path/ubm-finetune/wav.path.finetune.'+mode, 'r') as f:
					self.wavlist = f.readlines()
					for k, line in enumerate(self.wavlist):
						line = line.strip('\n')
						self.wavlist[k] = line

				with open(self.root_dir+'data_path/ubm-finetune/txt.path.finetune.'+mode, 'r') as g:
					self.txtlist = g.readlines()
					for k, line in enumerate(self.txtlist):
						line = line.strip('\n')
						self.txtlist[k] = line

	def __len__(self):
		assert len(self.wavlist) == len(self.txtlist)
		return len(self.wavlist)

	def __getitem__(self, idx):
		spk_id = self.wavlist[idx][-12:-8]
		spec_saved_indicator = False if self.spec_dir is None else os.path.exists(self.spec_dir+self.wavlist[idx][-17:-4]+'_mel.npy')

		if self.step in ['train_text2mel', 'synthesize']:
			if spec_saved_indicator:
				reduced_mel_spec = np.load(self.spec_dir+self.wavlist[idx][-17:-4]+'_mel.npy')
				if self.step == 'synthesize':
					lin_spec_norm = np.load(self.spec_dir+self.wavlist[idx][-17:-4]+'_lin.npy')

			else:
				speech, sr = librosa.core.load(path=self.wavlist[idx], sr=None, mono=True)
				speech, _ = librosa.effects.trim(speech, 22)
				speech = np.append(speech[0], speech[1:] - self.preemph*speech[:-1])
				lin_spec = np.abs(librosa.stft(y=speech, n_fft=self.n_fft, hop_length=self.hop))
				mel_filterbank = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.freq_bins)
				mel_spec = np.dot(mel_filterbank, lin_spec)

				if self.cfg['LOG_FEATURE']:
					mel_spec = 20*np.log10(np.maximum(1e-5, mel_spec))
					lin_spec = 20*np.log10(np.maximum(1e-5, lin_spec))
					mel_spec_norm = np.clip((mel_spec-self.cfg['REF_DB']+self.cfg['MAX_DB'])/self.cfg['MAX_DB'], 1e-8, 1)
					lin_spec_norm = np.clip((lin_spec-self.cfg['REF_DB']+self.cfg['MAX_DB'])/self.cfg['MAX_DB'], 1e-8, 1)

				else:
					maxlin = np.max(lin_spec)
					maxmel = np.max(mel_spec)

					lin_spec_norm = (lin_spec / maxlin)**self.norm_power
					mel_spec_norm = (mel_spec / maxmel)**self.norm_power

				# Downsampling along time frames.
				reduced_total_time = np.shape(mel_spec)[1] // self.tres_dsmp
				reduced_time = [self.tres_dsmp*k for k in range(reduced_total_time)]
				reduced_mel_spec = mel_spec_norm[:, reduced_time]
				lin_spec_norm = lin_spec_norm[:, :self.tres_dsmp*reduced_total_time]

				if self.spec_dir is not None:
					os.system('mkdir -p '+self.spec_dir+spk_id)
					np.save(self.spec_dir+self.wavlist[idx][-17:-4]+'_mel.npy', reduced_mel_spec)
					np.save(self.spec_dir+self.wavlist[idx][-17:-4]+'_lin.npy', lin_spec_norm)
						
			spk_emb = torch.from_numpy(np.expand_dims(np.load(self.spkemb_dir+spk_id+'.npy'), axis=1))

			with open(self.txtlist[idx], 'r') as f:
				text = f.readlines()
			text = text[0].strip()
			text = torch.from_numpy(text2id(text, self.vocabulary, self.char2idx))
			sample = {'data_0': torch.from_numpy(reduced_mel_spec), 'data_1': text, 'data_2': spk_emb} \
					 if (self.step == 'train_text2mel' or self.mode == 'validate') else {'data_0': torch.from_numpy(reduced_mel_spec), 'data_1': text, 'data_2': spk_emb, 'data_3': torch.from_numpy(lin_spec_norm)}

		if self.step == 'train_ssrn':
			if spec_saved_indicator:
				reduced_mel_spec = np.load(self.spec_dir+self.wavlist[idx][-17:-4]+'_mel.npy')
				lin_spec_norm = np.load(self.spec_dir+self.wavlist[idx][-17:-4]+'_lin.npy')

			else:
				speech, sr = librosa.core.load(path=self.wavlist[idx], sr=None, mono=True)
				speech, _ = librosa.effects.trim(speech, 22)
				speech = np.append(speech[0], speech[1:] - self.preemph*speech[:-1])
				lin_spec = np.abs(librosa.stft(y=speech, n_fft=self.n_fft, hop_length=self.hop))
				mel_filterbank = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.freq_bins)
				mel_spec = np.dot(mel_filterbank, lin_spec)

				if self.cfg['LOG_FEATURE']:
					mel_spec = 20*np.log10(np.maximum(1e-5, mel_spec))
					lin_spec = 20*np.log10(np.maximum(1e-5, lin_spec))
					mel_spec_norm = np.clip((mel_spec-self.cfg['REF_DB']+self.cfg['MAX_DB'])/self.cfg['MAX_DB'], 1e-8, 1)
					lin_spec_norm = np.clip((lin_spec-self.cfg['REF_DB']+self.cfg['MAX_DB'])/self.cfg['MAX_DB'], 1e-8, 1)

				else:
					maxlin = np.max(lin_spec)
					maxmel = np.max(mel_spec)

					lin_spec_norm = (lin_spec / maxlin)**self.norm_power
					mel_spec_norm = (mel_spec / maxmel)**self.norm_power

				# Downsampling along time frames.
				reduced_total_time = np.shape(mel_spec)[1] // self.tres_dsmp
				reduced_time = [self.tres_dsmp*k for k in range(reduced_total_time)]
				reduced_mel_spec = mel_spec_norm[:, reduced_time]
				lin_spec_norm = lin_spec_norm[:, :self.tres_dsmp*reduced_total_time]

				if self.spec_dir is not None:
					os.system('mkdir -p '+self.spec_dir+spk_id)
					np.save(self.spec_dir+self.wavlist[idx][-17:-4]+'_mel.npy', reduced_mel_spec)
					np.save(self.spec_dir+self.wavlist[idx][-17:-4]+'_lin.npy', lin_spec_norm)

			sample = {'data_0': torch.from_numpy(reduced_mel_spec), 'data_1': torch.from_numpy(lin_spec_norm)}

		return sample

def text2id(text, vocab, char2idx):
	# text = sample['data_1'] # string
	text = text.lower() + 'E' # Covert to lower case and append 'EOF' token.

	# Suppose no digits exists.
	tid = [char2idx[ch] if ch in vocab else None for ch in text]
	while None in tid:
		tid.remove(None)
	tid = np.expand_dims(np.array(tid), axis=0) # numpy array: 1*text_len

	return tid

def collate_pad_2(batch):
	"""
	Zero-padding to make the text and spectrogram have the same length.
	Note: As 'P'(denotes padding) pairs with 0 in the vocabulary, we
		  automatically use zero padding behind the trial of the text id.
	"""
	mel_max_length = max([batch[i]['data_0'].size()[1] for i in range(len(batch))])
	lin_max_length = max([batch[i]['data_1'].size()[1] for i in range(len(batch))])

	for i in range(len(batch)):
		len_diff = mel_max_length - batch[i]['data_0'].size()[1]
		if len_diff > 0:
			batch[i]['data_0'] = torch.cat((batch[i]['data_0'], torch.zeros((batch[i]['data_0'].size()[0], len_diff))), dim=1)
		len_diff = lin_max_length - batch[i]['data_1'].size()[1]
		if len_diff > 0:
			batch[i]['data_1'] = torch.cat((batch[i]['data_1'], torch.zeros((batch[i]['data_1'].size()[0], len_diff))), dim=1)

	batch_0 = torch.stack([batch[i]['data_0'] for i in range(len(batch))], dim=0)
	batch_1 = torch.stack([batch[i]['data_1'] for i in range(len(batch))], dim=0)

	return {'data_0': batch_0, 'data_1': batch_1}

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
			batch[i]['data_1'] = torch.cat((batch[i]['data_1'], torch.zeros((1, len_diff)).long()), dim=1)
		len_diff = spec_max_length - batch[i]['data_0'].size()[1]
		if len_diff > 0:
			batch[i]['data_0'] = torch.cat((batch[i]['data_0'], torch.zeros((batch[i]['data_0'].size()[0], len_diff))), dim=1)

	batch_0 = torch.stack([batch[i]['data_0'] for i in range(len(batch))], dim=0)
	batch_1 = torch.stack([batch[i]['data_1'] for i in range(len(batch))], dim=0)
	batch_2 = torch.stack([batch[i]['data_2'] for i in range(len(batch))], dim=0)

	return {'data_0': batch_0, 'data_1': batch_1, 'data_2': batch_2}

def collate_pad_4(batch):
	"""
	Zero-padding to make the text and spectrogram have the same length.
	Note: As 'P'(denotes padding) pairs with 0 in the vocabulary, we
		  automatically use zero padding behind the trial of the text id.
	"""
	text_max_length = max([batch[i]['data_1'].size()[1] for i in range(len(batch))])
	mel_max_length = max([batch[i]['data_0'].size()[1] for i in range(len(batch))])
	lin_max_length = max([batch[i]['data_3'].size()[1] for i in range(len(batch))])

	for i in range(len(batch)):
		len_diff = text_max_length - batch[i]['data_1'].size()[1]
		if len_diff > 0:
			batch[i]['data_1'] = torch.cat((batch[i]['data_1'], torch.zeros((1, len_diff)).long()), dim=1)
		len_diff = mel_max_length - batch[i]['data_0'].size()[1]
		if len_diff > 0:
			batch[i]['data_0'] = torch.cat((batch[i]['data_0'], torch.zeros((batch[i]['data_0'].size()[0], len_diff))), dim=1)
		len_diff = lin_max_length - batch[i]['data_3'].size()[1]
		if len_diff > 0:
			batch[i]['data_3'] = torch.cat((batch[i]['data_3'], torch.zeros((batch[i]['data_3'].size()[0], len_diff))), dim=1)

	batch_0 = torch.stack([batch[i]['data_0'] for i in range(len(batch))], dim=0)
	batch_1 = torch.stack([batch[i]['data_1'] for i in range(len(batch))], dim=0)
	batch_2 = torch.stack([batch[i]['data_2'] for i in range(len(batch))], dim=0)
	batch_3 = torch.stack([batch[i]['data_3'] for i in range(len(batch))], dim=0)

	return {'data_0': batch_0, 'data_1': batch_1, 'data_2': batch_2, 'data_3': batch_3}