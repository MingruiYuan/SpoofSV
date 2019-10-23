import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from models.TTSModel import melSyn, SSRN
from data.dataset import dataset, collate_pad_2, collate_pad_3, collate_pad_4
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# He's Gaussian initialization
def init_weights(layer):
	if hasattr(layer, 'weight'):
		if len(layer.weight.shape) > 1:
			torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def guided_attention_mat(max_text_len, max_frame_num):
	g = 0.2
	W = torch.zeros((max_text_len, max_frame_num))
	for k1 in range(max_text_len):
		for k2 in range(max_frame_num):
			W[k1, k2] = 1-math.exp(-(k2/max_frame_num-k1/max_text_len)**2/(2*g*g))
	W = W.to(device)
	return W

def plot_attention(att, iters, fig_dir):
	if torch.cuda.is_available():
		att = att.cpu()
	att = att.detach().numpy()

	if not os.path.exists(fig_dir):
		os.system('mkdir -p '+fig_dir)

	fig, ax = plt.subplots()
	img = ax.imshow(att)

	fig.colorbar(img)
	plt.title('{} iterations'.format(str(iters)))
	plt.savefig(fig_dir+'att_iteration_{}.png'.format(str(iters)), format='png')
	plt.close(fig)

def validate(loader, trainloader, gaw, cfg, model, train_step='train_text2mel'):
	with torch.no_grad():
		loss_avg = 0
		for i, sp in enumerate(loader):
			mel_gt = sp['data_0'].to(device)
			if train_step == 'train_text2mel':
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)
			if train_step == 'train_ssrn':
				lin_gt = sp['data_1'].to(device)
			
			# Should we use all 0 frame as the init frame?
			if train_step == 'train_text2mel':
				d1, d2, d3 = mel_gt.shape
				init_frame = torch.zeros((d1, d2, 1)).to(device)
				Y, A, prev_maxatt, K, V = model(melspec=init_frame, textid=text_id, spkemb=spk_emb, pma=torch.zeros((d1,)).long().to(device))
				inputs = torch.cat((init_frame, Y), dim=-1)
				for frame in range(d3-1):
					Y, A, prev_maxatt = model(melspec=inputs, textid=None, spkemb=spk_emb, K=K, V=V, A_last=A, pma=prev_maxatt)
					inputs = torch.cat((inputs, Y[:, :, -1:]), dim=-1)

				# loss: Y with mel_gt and Attention.
				loss_l1 = torch.mean(torch.abs(mel_gt-Y))
				loss_bin_div = torch.mean(-mel_gt*torch.log(Y+1e-8)-(1-mel_gt)*torch.log(1-Y+1e-8))
				A_aug = F.pad(A, (0, cfg['MAX_FRAME_NUM']-A.size()[-1], 0, cfg['MAX_TEXT_LEN']-A.size()[-2]), value=-1)
				# Here gaw will broadcast along axis 0.
				loss_att = torch.sum(torch.ne(A_aug, -1).float()*A_aug*gaw) / torch.sum(torch.ne(A_aug, -1).float())

				loss = loss_l1 + loss_bin_div + loss_att
				print('val set loss: {} {} {} {}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss.item())))

			# continue SSRN with Y(predicted melspec) and lin_gt
			if train_step == 'train_ssrn':
				pred_lin_prob = model(mel_gt)
				loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
				loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))

				loss = loss_l1 + loss_bin_div
				print('val set loss: {} {} {}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss.item())))

			loss_avg += loss.item()

		# validate on a batch from trainset.
		for i, sp in enumerate(trainloader):
			mel_gt = sp['data_0'].to(device)
			if train_step == 'train_text2mel':
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)
			if train_step == 'train_ssrn':
				lin_gt = sp['data_1'].to(device)

			if train_step == 'train_text2mel':
				d1, d2, d3 = mel_gt.shape
				init_frame = torch.zeros((d1, d2, 1)).to(device)
				Y, A, prev_maxatt, K, V = model(melspec=init_frame, textid=text_id, spkemb=spk_emb, pma=torch.zeros((d1,)).long().to(device))
				inputs = torch.cat((init_frame, Y), dim=-1)
				for frame in range(d3-1):
					Y, A, prev_maxatt = model(melspec=inputs, textid=None, spkemb=spk_emb, K=K, V=V, A_last=A, pma=prev_maxatt)
					inputs = torch.cat((inputs, Y[:, :, -1:]), dim=-1)

				# loss: Y with mel_gt and Attention.
				loss_l1 = torch.mean(torch.abs(mel_gt-Y))
				loss_bin_div = torch.mean(-mel_gt*torch.log(Y+1e-8)-(1-mel_gt)*torch.log(1-Y+1e-8))
				A_aug = F.pad(A, (0, cfg['MAX_FRAME_NUM']-A.size()[-1], 0, cfg['MAX_TEXT_LEN']-A.size()[-2]), value=-1)
				# Here gaw will broadcast along axis 0.
				loss_att = torch.sum(torch.ne(A_aug, -1).float()*A_aug*gaw) / torch.sum(torch.ne(A_aug, -1).float())

				loss = loss_l1 + loss_bin_div + loss_att
				print('train set loss: {} {} {} {}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss.item())))

			# continue SSRN with Y(predicted melspec) and lin_gt
			if train_step == 'train_ssrn':
				pred_lin_prob = model(mel_gt)
				loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
				loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))

				loss = loss_l1 + loss_bin_div
				print('train set loss: {} {} {}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss.item())))

			break		

		return loss_avg/len(loader), loss.item()


def ordinary_train(train_step, 
		  		   train_pattern,
		           cfg,
		           spec_dir=None,
		           resume_checkpoints=None,
		           current_time=None):
	"""
	Args:
	train_step: 'train_text2mel' or 'train_ssrn'.
	train_pattern: 'universal', 'conditional'. 'ubm-finetune' should be in another function.
	resume_checkpoints: directory of checkpoints to resume.
	"""
	checkpoints_dir = cfg['SRC_ROOT_DIR'] + 'checkpoints/'
	current_save_dir = checkpoints_dir+train_pattern+'/not_adversarial/'+current_time
	fig_dir = current_save_dir+'/fig/'
	if not os.path.exists(current_save_dir):
		os.system('mkdir -p '+current_save_dir)

	if cfg['APPLY_DROPOUT']:
		from models.TTSModel_dropout import melSyn, SSRN
	else:
		from models.TTSModel import melSyn, SSRN

	train_dataset = dataset(cfg=cfg, mode='train', pattern=train_pattern, step=train_step, spec_dir=spec_dir)

	validate_dataset = dataset(cfg=cfg, mode='validate', pattern=train_pattern, step=train_step, spec_dir=spec_dir)

	if train_step == 'train_text2mel':
		# subtract 1 because we merge "'" and '"'.
		model = melSyn(vocab_len=len(cfg['VOCABULARY'])-1,
			           condition = (train_pattern == 'conditional'),
			           spkemb_dim=cfg['SPK_EMB_DIM'],
			           textemb_dim=cfg['TEXT_EMB_DIM'],
			           freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
			           hidden_dim=cfg['HIDDEN_DIM'])
		if cfg['MULTI_GPU']:
			model = torch.nn.DataParallel(model)

	if train_step == 'train_ssrn':
		model = SSRN(freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
					 output_bins=(1+cfg['STFT']['FFT_LENGTH']//2),
					 ssrn_dim=cfg['SSRN_DIM'])
		if cfg['MULTI_GPU']:
			model = torch.nn.DataParallel(model)

	# If train from scratch, initialize recursively.
	if resume_checkpoints is None:
		model.apply(init_weights)
		epoch = 0
		iteration = 0
		print('CUDA available: ', torch.cuda.is_available())
		model.to(device)
		optimizer = optim.Adam(model.parameters(), cfg['ADAM']['ALPHA'], (cfg['ADAM']['BETA_1'], cfg['ADAM']['BETA_2']), cfg['ADAM']['EPSILON'])
		# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=1, verbose=True, min_lr=1e-8)
		loss_val_log = []
	else:
		# load checkpoint.
		print('CUDA available: ', torch.cuda.is_available())
		checkpoint = torch.load(resume_checkpoints)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.to(device)
		optimizer = optim.Adam(model.parameters(), cfg['ADAM']['ALPHA'], (cfg['ADAM']['BETA_1'], cfg['ADAM']['BETA_2']), cfg['ADAM']['EPSILON'])

		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		epoch = checkpoint['epoch']
		iteration = checkpoint['iteration']
		loss_val_log = checkpoint['loss_val_log']

	train_loader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=collate_pad_3 if train_step == 'train_text2mel' else collate_pad_2)
	validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_pad_3 if train_step == 'train_text2mel' else collate_pad_2)

	# guided attention weights.
	gaw = guided_attention_mat(cfg['MAX_TEXT_LEN'], cfg['MAX_FRAME_NUM'])

	# check model's device
	print('Model device:', next(model.parameters()).device)
	loss_iter = 0

	while epoch < cfg['MAX_EPOCHS']:
		#start_epoch = time.time()
		print('Epoch ', epoch+1)
		print('*******************')
		# loss_epoch = 0
		loader_len = len(train_loader)

		for i, sp in enumerate(train_loader):
			# Training.
			start_iter = time.time()
			optimizer.zero_grad()

			if train_step == 'train_text2mel':
				mel_gt = sp['data_0'].to(device)
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)

				spec_inputs = torch.cat((torch.zeros_like(mel_gt[:,:,:1]), mel_gt[:, :, :-1]), dim=-1)
				pred_mel_prob, att_mat = model(spec_inputs, text_id, spk_emb)

				# Loss.
				loss_l1 = torch.mean(torch.abs(mel_gt-pred_mel_prob))
				loss_bin_div = torch.mean(-mel_gt*torch.log(pred_mel_prob+1e-8)-(1-mel_gt)*torch.log(1-pred_mel_prob+1e-8))
				att_aug = F.pad(att_mat, (0, cfg['MAX_FRAME_NUM']-att_mat.size()[-1], 0, cfg['MAX_TEXT_LEN']-att_mat.size()[-2]), value=-1)
				# Here gaw will broadcast along axis 0.
				loss_att = torch.sum(torch.ne(att_aug, -1).float()*att_aug*gaw) / torch.sum(torch.ne(att_aug, -1).float())

				loss = loss_l1 + loss_bin_div + loss_att
				loss.backward()
				optimizer.step()
				loss_iter += loss.item()
				print('Iteration {}/{}'.format(str(i+1), str(loader_len)), ' for epoch {}, loss: {} {} {} {}'.format(str(epoch+1), str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss.item())), 'global iteration {}'.format(str(iteration+1)))

			if train_step == 'train_ssrn':
				mel_gt = sp['data_0'].to(device)
				lin_gt = sp['data_1'].to(device)

				pred_lin_prob = model(mel_gt)

				# Loss.
				loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
				loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))

				loss = loss_l1 + loss_bin_div
				loss.backward()
				optimizer.step()
				loss_iter += loss.item()
				print('Iteration {}/{}'.format(str(i+1), str(loader_len)), ' for epoch {}, loss: {} {} {}'.format(str(epoch+1), str(loss_l1.item()), str(loss_bin_div.item()), str(loss.item())), 'global iteration {}'.format(str(iteration+1)))

			if (iteration % cfg['VAL_EVERY_ITER'] == 0) and iteration > 0:
				print('\n')
				print('No.{} VALIDATION'.format(str(iteration//cfg['VAL_EVERY_ITER'])))
				print('Average training loss: ', loss_iter/cfg['VAL_EVERY_ITER'])
				loss_iter = 0

				model.eval()
				loss_val, loss_val_train = validate(loader=validate_loader, trainloader=train_loader, gaw=gaw, cfg=cfg, model=model, train_step=train_step)
				loss_val_log.append(loss_val)
				model.train()

				if loss_val_log.index(min(loss_val_log)) == len(loss_val_log)-1:
					print('Current Best Model!')
					torch.save({'epoch': epoch+1,
								'iteration': iteration+1,
								'model_state_dict': model.module.state_dict() if cfg['MULTI_GPU'] else model.state_dict(),
								'optimizer_state_dict': optimizer.state_dict(),
								'loss_val_log': loss_val_log}, current_save_dir+'/{}.tar.pth'.format(train_step[6:]+'_best_model'))

				print('Validation loss of No.{} validation: {} on validation set. {} on train set.'.format(str(iteration//cfg['VAL_EVERY_ITER']), str(loss_val), str(loss_val_train)))			

				torch.save({'epoch': epoch+1,
							'iteration': iteration+1,
						    'model_state_dict': model.module.state_dict() if cfg['MULTI_GPU'] else model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'loss_val_log': loss_val_log}, current_save_dir+'/{}_iteration_{}.tar.pth'.format(train_step[6:], str(iteration+1)))
				print('At iteration ', iteration+1, '{} model saved at {}'.format(train_step[6:], current_save_dir))

				if train_step == 'train_text2mel':
					plot_attention(att=att_mat[0, :, :], iters=iteration+1, fig_dir=fig_dir)

			end_iter = time.time()
			iteration += 1 
			print('Time elapsed {}s'.format(str(end_iter-start_iter)))

		epoch += 1