import os
import random
import argparse
import torch
import librosa
import soundfile as sf
from models.TTSModel import melSyn, SSRN
import json
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text2id(text, vocab, char2idx):
	# text = sample['data_1'] # string
	text = text.lower() + 'E' # Covert to lower case and append 'EOF' token.

	# Suppose no digits exists.
	tid = [char2idx[ch] if ch in vocab else None for ch in text]
	while None in tid:
		tid.remove(None)
	tid = np.expand_dims(np.array(tid), axis=0) # numpy array: 1*text_len

	return tid

# def plot_attention(att, spk, fig_dir):
# 	if torch.cuda.is_available():
# 		att = att.cpu()
# 	att = att.numpy()

# 	if not os.path.exists(fig_dir):
# 		os.system('mkdir -p '+fig_dir)

# 	fig, ax = plt.subplots()
# 	img = ax.imshow(att)

# 	fig.colorbar(img)
# 	plt.title('Sample from speaker {}'.format(str(spk)))
# 	plt.savefig(fig_dir+'att_speaker_{}.png'.format(str(spk)), format='png')
# 	plt.close(fig)


if __name__ == '__main__':
	ps = argparse.ArgumentParser(description='Adversarial Conditional Text-to-speech')
	ps.add_argument('-C', '--configuration', type=str, default=None)
	ps.add_argument('--train_spk_num', type=int, default=88)
	ps.add_argument('--enroll_utt_num', type=int, default=3)
	ps.add_argument('--eval_utt_num', type=int, default=20)
	ps.add_argument('-T', '--current_time', type=str, required=True)
	args = ps.parse_args()

	with open(args.configuration, 'r') as f:
		cfg = json.load(f)

	with open(cfg['TTS_TEXTS'], 'r') as f:
		hs = f.readlines()
		for k, sen in enumerate(hs):
			hs[k] = sen.strip()

	char2idx = {char: idx for idx, char in enumerate(cfg['VOCABULARY'])}
	char2idx['"'] = len(cfg['VOCABULARY'])-2
	texts = []
	for k in range(args.eval_utt_num):
		text_id = text2id(hs[k], cfg['VOCABULARY'], char2idx)
		text_id = torch.from_numpy(text_id).to(device)
		texts.append(text_id)
	max_len = max([k.shape[-1] for k in texts])
	for k in range(len(texts)):
		len_diff = max_len - texts[k].shape[-1]
		if len_diff > 0:
			texts[k] = torch.cat((texts[k], torch.zeros((1, len_diff)).long().to(device)), dim=-1)
	text_id = torch.stack([k for k in texts], dim=0)

	m1 = melSyn(vocab_len=len(cfg['VOCABULARY'])-1,
				condition = True,
				spkemb_dim=cfg['SPK_EMB_DIM'],
				textemb_dim=cfg['TEXT_EMB_DIM'],
				freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
				hidden_dim=cfg['HIDDEN_DIM'])

	m2 = SSRN(freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
			  output_bins=(1+cfg['STFT']['FFT_LENGTH']//2),
			  ssrn_dim=cfg['SSRN_DIM'])	

	ckp1 = torch.load(cfg['INFERENCE_TEXT2MEL_MODEL'])
	ckp2 = torch.load(cfg['INFERENCE_SSRN_MODEL'])
	m1.load_state_dict(ckp1['model_state_dict'])
	m1.to(device)
	m1.eval()
	m2.load_state_dict(ckp2['model_state_dict'])
	m2.to(device)
	m2.eval()

	test_root_dir = cfg['SRC_ROOT_DIR'] + 'test/'
	save_dir = test_root_dir + args.current_time + '/spoof_data/'

	with torch.no_grad():
		spk_list = os.listdir(cfg['DATA_ROOT_DIR'] + 'wav22/')
		for spk in spk_list:
			print('Speaker ', spk)
			if not os.path.exists(save_dir+'s'+spk[1:]):
				os.system('mkdir -p '+save_dir+'s'+spk[1:])

			init_frame = torch.zeros((args.eval_utt_num, cfg['COARSE_MELSPEC']['FREQ_BINS'], 1)).to(device)
			spk_emb = torch.stack(args.eval_utt_num*[torch.from_numpy(np.expand_dims(np.load(cfg['SPK_EMB_DIR']+spk+'.npy'), axis=1)).to(device)], dim=0)

			Y, A, prev_maxatt, K, V = m1(melspec=init_frame, textid=text_id, spkemb=spk_emb, pma=torch.zeros((args.eval_utt_num,)).long().to(device))
			inputs = torch.cat((init_frame, Y), dim=-1)
			cnt = 1
			while True:
				Y, A, prev_maxatt = m1(melspec=inputs, textid=None, spkemb=spk_emb, K=K, V=V, A_last=A, pma=prev_maxatt)
				inputs = torch.cat((inputs, Y[:, :, -1:]), dim=-1)
				if cnt == cfg['MAX_FRAME_NUM']:
					break
				cnt += 1

			# plot_attention(att=A[0, :, :], spk=spk[1:], fig_dir=fig_dir)

			pred_lin_prob = m2(Y)

			if torch.cuda.is_available():
				pred_lin_prob = pred_lin_prob.cpu()
			pred_lin = pred_lin_prob.numpy()

			if cfg['LOG_FEATURE']:
				pred_lin = pred_lin*cfg['MAX_DB'] - cfg['MAX_DB'] + cfg['REF_DB']
				pred_lin = np.power(10, 0.05*pred_lin)

			for k in range(args.eval_utt_num):
				if not cfg['LOG_FEATURE']:
					pred_lin[k, :, :] = pred_lin[k, :, :]/np.max(pred_lin[k, :, :])
				spec = pred_lin[k, :, :]**(cfg['NORM_POWER']['RECONSTRUCTION']/cfg['NORM_POWER']['ANALYSIS'])
				time_signal = librosa.core.griffinlim(S=spec, n_iter=64, hop_length=cfg['STFT']['HOP_LENGTH'], win_length=cfg['STFT']['FFT_LENGTH'])
				time_signal = signal.lfilter([1], [1, -cfg['PREEMPH']], time_signal)
				time_signal, _ = librosa.effects.trim(time_signal, 30)
				if len(time_signal) > 9*cfg['SAMPLING_RATE']:
					time_signal = time_signal[:9*cfg['SAMPLING_RATE']]
				librosa.output.write_wav(save_dir+'s'+spk[1:]+'/s{}_{}.wav'.format(spk[1:], str(k+1).zfill(3)), time_signal if cfg['LOG_FEATURE'] else time_signal/np.max(time_signal)*0.75, cfg['SAMPLING_RATE'])

	# if args.sv_model == 'ivector':
	ivector_data_root = test_root_dir + args.current_time + '/ivector_data/'
	txt_root = cfg['DATA_ROOT_DIR'] + 'txt/'
	real_data_root = cfg['DATA_ROOT_DIR'] + 'wav22/'
	real_data_list = os.listdir(real_data_root)
	real_data_list.sort(key = lambda x:x[:])
	syn_data_root = save_dir
	syn_data_list = os.listdir(syn_data_root)
	syn_data_list.sort(key = lambda x:x[:])

	if not os.path.exists(ivector_data_root + 'transcript/'):
		os.system('mkdir -p '+ ivector_data_root + 'transcript/')
	ivector_transcript = open(ivector_data_root +'transcript/VCTK-transcript.txt', 'w')
	ivector_transcript_nospoof = open(ivector_data_root+'VCTK-transcript_nospoof.txt', 'w')

	print('I-VECTORs generation')
	for i, spk in enumerate(real_data_list):
		print(i, spk[1:])
		assert spk[1:] == syn_data_list[i][1:]
		if i < args.train_spk_num:
			if not os.path.exists(ivector_data_root+'wav/train/'+spk[1:]):
				os.system('mkdir -p '+ivector_data_root+'wav/train/'+spk[1:])

			utt_list = os.listdir(real_data_root+spk)
			random.shuffle(utt_list)
			for j in range(len(utt_list)):
				wavname_old = real_data_root + spk + '/' + utt_list[j]
				wavname_new = ivector_data_root + 'wav/train/' + spk[1:] + '/' + spk[1:] + 'W' + str(j+1).zfill(3) + '.wav'
				os.system('cp {} {}'.format(wavname_old, wavname_new))

				file_realspk = open(txt_root+spk+'/'+utt_list[j][:-4]+'.txt', 'r')
				line = file_realspk.readline()
				line = line.strip()
				ivector_transcript.write(spk[1:] + 'W' + str(j+1).zfill(3) + '    ' + line + '\n')
				ivector_transcript_nospoof.write(spk[1:] + 'W' + str(j+1).zfill(3) + '    ' + line + '\n')
				file_realspk.close()

			if i == 0:
				if not os.path.exists(ivector_data_root+'wav/dev/'):
					os.system('mkdir -p '+ivector_data_root+'wav/dev/')
				os.system('cp -r {} {}'.format(ivector_data_root+'wav/train/'+spk[1:], ivector_data_root+'wav/dev/'))

		else:
			if not os.path.exists(ivector_data_root+'wav/test/'+spk[1:]):
				os.system('mkdir -p '+ivector_data_root+'wav/test/'+spk[1:])
			if not os.path.exists(ivector_data_root+'test_nospoof/'+spk[1:]):
				os.system('mkdir -p '+ivector_data_root+'test_nospoof/'+spk[1:])

			utt_list = os.listdir(real_data_root+spk)
			random.shuffle(utt_list)
			for j in range(args.enroll_utt_num+args.eval_utt_num):
				wavname_old = real_data_root + spk + '/' + utt_list[j]
				wavname_new = ivector_data_root + 'wav/test/' + spk[1:] + '/' + spk[1:] + 'W' + str(j+1).zfill(3) + '.wav'
				wavname_new_nospoof = ivector_data_root + 'test_nospoof/' + spk[1:] + '/' + spk[1:] + 'W' + str(j+1).zfill(3) + '.wav'
				os.system('cp {} {}'.format(wavname_old, wavname_new))
				os.system('cp {} {}'.format(wavname_old, wavname_new_nospoof))

				file_realspk = open(txt_root+spk+'/'+utt_list[j][:-4]+'.txt', 'r')
				line = file_realspk.readline()
				line = line.strip()
				ivector_transcript.write(spk[1:] + 'W' + str(j+1).zfill(3) + '    ' + line + '\n')
				ivector_transcript_nospoof.write(spk[1:] + 'W' + str(j+1).zfill(3) + '    ' + line + '\n')
				file_realspk.close()

			syn_list = os.listdir(syn_data_root+'s'+spk[1:])
			syn_list.sort(key = lambda x:x[:-4])

			# Be sure to make kaldi script compatible.
			for j in range(args.eval_utt_num):
				wavname_old = syn_data_root + 's' + spk[1:] + '/' + syn_list[j]
				wavname_new = ivector_data_root + 'wav/test/' + spk[1:] + '/' + spk[1:] + 'W' + str(j+args.eval_utt_num+args.enroll_utt_num+1).zfill(3) + '.wav'
				os.system('cp {} {}'.format(wavname_old, wavname_new))

				ivector_transcript.write(spk[1:] + 'W' + str(j+args.eval_utt_num+args.enroll_utt_num+1).zfill(3) + '    ' + hs[j] + '\n')

	ivector_transcript.close()
	ivector_transcript_nospoof.close()
	
	ge2e_dir = test_root_dir + args.current_time + '/ge2e_data/'
	if not os.path.exists(ge2e_dir):
		os.system('mkdir -p '+ge2e_dir)
	src_train = ivector_data_root + 'wav/train/*'
	src_test = ivector_data_root + 'wav/test/*'
	os.system('ln -s {} {}'.format(src_train, ge2e_dir))
	os.system('ln -s {} {}'.format(src_test, ge2e_dir))

	antispoof_savedir = cfg['ANTISPOOF_DIR'] + args.current_time + '/flac/'
	if not os.path.exists(antispoof_savedir):
		os.system('mkdir -p '+antispoof_savedir)
	protocol = open(cfg['ANTISPOOF_DIR']+'ASVspoof2019_LA_cm_protocols/customized_data_{}.txt'.format(args.current_time), 'w')
	bonafide_num = 10*108

	with open(cfg['ANTISPOOF_DIR']+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'r') as f:
		dev_proto = f.readlines()
	index = 0

	for k in range(bonafide_num):
		info = dev_proto[index].strip().split()
		assert info[-1] == 'bonafide'
		audio_location = cfg['ANTISPOOF_DIR'] + 'ASVspoof2019_LA_dev/flac/' + info[1] + '.flac'
		save_location = antispoof_savedir + 'LA_D_{}.flac'.format(str(index+1).zfill(7))
		os.system('cp {} {}'.format(audio_location, save_location))
		protocol.write('{} LA_D_{} - - bonafide\n'.format(info[0], str(index+1).zfill(7)))
		index += 1

	spoof_spk_list = os.listdir(save_dir)
	for i, spk in enumerate(spoof_spk_list):
		spoof_utt_list = os.listdir(save_dir+spk)
		for j, utt in enumerate(spoof_utt_list):
			audio_location = save_dir + spk + '/' + utt
			audio, sr = librosa.load(audio_location, sr=16000)
			assert sr==16000
			save_location = antispoof_savedir + 'LA_D_{}.flac'.format(str(index+1).zfill(7))
			sf.write(save_location, audio, samplerate=16000)
			protocol.write('{} LA_D_{} - - spoof\n'.format(spk, str(index+1).zfill(7)))
			index += 1

	protocol.close()