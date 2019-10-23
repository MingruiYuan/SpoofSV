import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import dataset, collate_pad_2, collate_pad_3, collate_pad_4
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# He's Gaussian initialization
def init_weights(layer):
	if hasattr(layer, 'weight'):
		if len(layer.weight.shape) > 1:
			torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def clip_weights(layer):
	if hasattr(layer, 'weight'):
		layer.weight.requires_grad = False
		torch.clamp_(layer.weight, min=-0.1, max=0.1)
		layer.weight.requires_grad = True

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

def plot_loss(losses, iters, fig_dir):
	if not os.path.exists(fig_dir):
		os.system('mkdir -p '+fig_dir)

	# fig1, ax1 = plt.subplots(2,1)
	# ax1[0].set_title('Generator Validation Loss')
	# ax1[1].set_title('Discriminator Validation Loss')
	# ax1[0].plot(losses['v_s'], color='blue')
	# ax1[0].plot(losses['v_s_o'], color='orange')
	# ax1[0].legend(['total', 'only from D'])
	# ax1[1].plot(losses['v_d'], color='green')
	# plt.savefig(fig_dir+'ValLoss_iteration_{}.png'.format(str(iters)), format='png')

	fig2, ax2 = plt.subplots(3,1)
	fig2.tight_layout()
	ax2[0].set_title('Generator Train Loss')
	ax2[1].set_title('Discriminator Train Loss')
	ax2[2].set_title('Wasserstein Distance')
	ax2[0].plot(losses['t_s'], color='blue')
	ax2[0].plot(losses['t_s_o'], color='orange')
	ax2[0].legend(['total', 'only from D'])
	ax2[1].plot(losses['t_d'], color='green')
	ax2[2].plot(losses['wd'], color='purple')
	plt.savefig(fig_dir+'TrainLoss_iteration_{}.png'.format(str(iters)), format='png')

	fig3, ax3 = plt.subplots(2,1)
	fig3.tight_layout()
	ax3[0].set_title('Generator Smooth Train Loss')
	ax3[1].set_title('Discriminator Smooth Train Loss')
	ax3[0].plot(losses['t_s_s'], color='blue')
	ax3[0].plot(losses['t_s_s_o'], color='orange')
	ax3[0].legend(['total', 'only from D'])
	ax3[1].plot(losses['t_s_d'], color='green')
	plt.savefig(fig_dir+'SmoothTrainLoss_iteration_{}.png'.format(str(iters)), format='png')

def validate(loader, trainloader, gaw, cfg, model, disc, train_step):
	with torch.no_grad():
		loss_avg_syn = 0
		loss_avg_syn_onlyfromD = 0
		loss_avg_disc = 0
		for i, sp in enumerate(loader):
			mel_gt = sp['data_0'].to(device)
			if train_step == 'train_text2mel':
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)
			if train_step == 'train_ssrn':
				lin_gt = sp['data_1'].to(device)

			if train_step == 'train_text2mel':
				spec_inputs = torch.cat((torch.zeros_like(mel_gt[:, :, :1]), mel_gt[:, :, :-1]), dim=-1)
				pred_mel_prob, att_mat = model(spec_inputs, text_id, spk_emb)
				disc_gt = disc(mel_gt)
				disc_syn = disc(pred_mel_prob)

				loss_l1 = torch.mean(torch.abs(mel_gt-pred_mel_prob))
				loss_bin_div = torch.mean(-mel_gt*torch.log(pred_mel_prob+1e-8)-(1-mel_gt)*torch.log(1-pred_mel_prob+1e-8))
				att_aug = F.pad(att_mat, (0, cfg['MAX_FRAME_NUM']-att_mat.size()[-1], 0, cfg['MAX_TEXT_LEN']-att_mat.size()[-2]), value=-1)
				# Here gaw will broadcast along axis 0.
				loss_att = torch.sum(torch.ne(att_aug, -1).float()*att_aug*gaw) / torch.sum(torch.ne(att_aug, -1).float())
				loss_disc = torch.mean(-torch.log(disc_syn+1e-8))
				loss_val_G = loss_l1 + loss_bin_div + loss_att + loss_disc

				loss_val_D = torch.mean(-torch.log(disc_gt+1e-8)-torch.log(1-disc_syn+1e-8))
				print('Val set loss: Discriminator: ', loss_val_D.item())
				print('Generator: L1:{}, BD:{}, ATT:{}, DISC:{}, ALL:{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss_disc.item()), str(loss_val_G.item())))

			if train_step == 'train_ssrn':
				pred_lin_prob = model(mel_gt)
				disc_gt = disc(lin_gt)	
				disc_syn = disc(pred_lin_prob)

				loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
				loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))
				loss_disc = torch.mean(-torch.log(disc_syn+1e-8))
				loss_val_G = loss_l1 + loss_bin_div + loss_disc

				loss_val_D = torch.mean(-torch.log(disc_gt+1e-8)-torch.log(1-disc_syn+1e-8))
				print('Val set loss: Discriminator: ', loss_val_D.item())
				print('Generator: L1:{}, BD:{}, DISC:{}, ALL:{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_disc.item()), str(loss_val_G.item())))

			loss_avg_syn += loss_val_G.item()
			loss_avg_syn_onlyfromD += loss_disc.item()
			loss_avg_disc += loss_val_D.item()

		for i, sp in enumerate(trainloader):
			mel_gt = sp['data_0'].to(device)
			if train_step == 'train_text2mel':
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)
			if train_step == 'train_ssrn':
				lin_gt = sp['data_1'].to(device)

			if train_step == 'train_text2mel':
				spec_inputs = torch.cat((torch.zeros_like(mel_gt[:, :, :1]), mel_gt[:, :, :-1]), dim=-1)
				pred_mel_prob, att_mat = model(spec_inputs, text_id, spk_emb)
				disc_gt = disc(mel_gt)
				disc_syn = disc(pred_mel_prob)

				loss_l1 = torch.mean(torch.abs(mel_gt-pred_mel_prob))
				loss_bin_div = torch.mean(-mel_gt*torch.log(pred_mel_prob+1e-8)-(1-mel_gt)*torch.log(1-pred_mel_prob+1e-8))
				att_aug = F.pad(att_mat, (0, cfg['MAX_FRAME_NUM']-att_mat.size()[-1], 0, cfg['MAX_TEXT_LEN']-att_mat.size()[-2]), value=-1)
				# Here gaw will broadcast along axis 0.
				loss_att = torch.sum(torch.ne(att_aug, -1).float()*att_aug*gaw) / torch.sum(torch.ne(att_aug, -1).float())
				loss_disc = torch.mean(-torch.log(disc_syn+1e-8))
				loss_train_G = loss_l1 + loss_bin_div + loss_att + loss_disc

				loss_train_D = torch.mean(-torch.log(disc_gt+1e-8)-torch.log(1-disc_syn+1e-8))
				print('Val set loss: Discriminator: ', loss_val_D.item())
				print('Generator: L1:{}, BD:{}, ATT:{}, DISC:{}, ALL:{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss_disc.item()), str(loss_val_G.item())))

			if train_step == 'train_ssrn':
				pred_lin_prob = model(mel_gt)
				disc_gt = disc(lin_gt)	
				disc_syn = disc(pred_lin_prob)

				loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
				loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))
				loss_disc = torch.mean(-torch.log(disc_syn+1e-8))
				loss_train_G = loss_l1 + loss_bin_div + loss_disc

				loss_train_D = torch.mean(-torch.log(disc_gt+1e-8)-torch.log(1-disc_syn+1e-8))
				print('Val set loss: Discriminator: ', loss_val_D.item())
				print('Generator: L1:{}, BD:{}, DISC:{}, ALL:{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_disc.item()), str(loss_val_G.item())))

			break

		return loss_avg_syn/len(loader), loss_avg_syn_onlyfromD/len(loader), loss_avg_disc/len(loader), loss_train_G.item(), loss_train_D.item()

def adversarial_train(train_step,
					  train_pattern,
					  cfg,
					  spec_dir=None,
					  checkpoints_dir=None,
					  resume_checkpoints=None,
					  is_parallel=False,
					  dropout=False,
					  current_time=None):

	current_save_dir = checkpoints_dir+train_pattern+'/adversarial/'+current_time
	fig_dir = current_save_dir+'/fig/'
	if not os.path.exists(current_save_dir):
		os.system('mkdir -p '+current_save_dir)

	if dropout:
		from models.TTSModel_dropout import melSyn, SSRN
	else:
		from models.TTSModel import melSyn, SSRN

	from models.discriminator import melDisc, linDisc

	train_dataset = dataset(root_dir=cfg['DATA_ROOT_DIR'],
				       		spkemb_dir=cfg['SPK_EMB_DIR'],
				       		vocabulary=cfg['VOCABULARY'],
				       		n_mels=cfg['COARSE_MELSPEC']['FREQ_BINS'],
				       		time_frame_reduction=cfg['COARSE_MELSPEC']['REDUCTION'],
				       		preemphasis=cfg['PREEMPH'],
				       		n_fft=cfg['STFT']['FFT_LENGTH'],
				       		hop=cfg['STFT']['HOP_LENGTH'],
				       		norm_power=cfg['NORM_POWER']['ANALYSIS'],
				      		mode='train',
				       		pattern=train_pattern,
				       		step=train_step,
				       		spec_dir=spec_dir)

	validate_dataset = dataset(root_dir=cfg['DATA_ROOT_DIR'],
				       		   spkemb_dir=cfg['SPK_EMB_DIR'],
				       		   vocabulary=cfg['VOCABULARY'],
				       		   n_mels=cfg['COARSE_MELSPEC']['FREQ_BINS'],
				       		   time_frame_reduction=cfg['COARSE_MELSPEC']['REDUCTION'],
				       		   preemphasis=cfg['PREEMPH'],
				       		   n_fft=cfg['STFT']['FFT_LENGTH'],
				       		   hop=cfg['STFT']['HOP_LENGTH'],
				       		   norm_power=cfg['NORM_POWER']['ANALYSIS'],
				      		   mode='validate',
				       		   pattern=train_pattern,
				       		   step=train_step,
				       		   spec_dir=spec_dir)

	if train_step == 'train_text2mel':
		# subtract 1 because we merge "'" and '"'.
		model = melSyn(vocab_len=len(cfg['VOCABULARY'])-1,
			           condition = (train_pattern == 'conditional'),
			           spkemb_dim=cfg['SPK_EMB_DIM'],
			           textemb_dim=cfg['TEXT_EMB_DIM'],
			           freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
			           hidden_dim=cfg['HIDDEN_DIM'])

		disc = melDisc(freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'], disc_dim=cfg['DISC_DIM'])

		if is_parallel:
			model = torch.nn.DataParallel(model)
			disc = torch.nn.DataParallel(disc)

	if train_step == 'train_ssrn':
		model = SSRN(freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
					 output_bins=(1+cfg['STFT']['FFT_LENGTH']//2),
					 ssrn_dim=cfg['SSRN_DIM'])

		disc = linDisc(freq_bins=(1+cfg['STFT']['FFT_LENGTH']//2), disc_dim=cfg['DISC_DIM'])

		if is_parallel:
			model = torch.nn.DataParallel(model)
			disc = torch.nn.DataParallel(disc)

	# If train from scratch, initialize recursively.
	if resume_checkpoints is None:
		model.apply(init_weights)
		disc.apply(init_weights)
		epoch = 0
		iteration = 0
		print('CUDA available: ', torch.cuda.is_available())
		model.to(device)
		disc.to(device)
		opt_syn = optim.RMSprop(model.parameters(), lr=cfg['ADAM']['ALPHA'], eps=1e-6)
		opt_disc = optim.RMSprop(disc.parameters(), lr=cfg['ADAM']['ALPHA'], eps=1e-6)
		# opt_syn = optim.Adam(model.parameters(), cfg['ADAM']['ALPHA'], (cfg['ADAM']['BETA_1'], cfg['ADAM']['BETA_2']), cfg['ADAM']['EPSILON'])
		# opt_disc = optim.Adam(disc.parameters(), cfg['ADAM']['ALPHA'], (cfg['ADAM']['BETA_1'], cfg['ADAM']['BETA_2']), cfg['ADAM']['EPSILON'])
		# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=1, verbose=True, min_lr=1e-8)
		# loss_val_log_syn = []
		# loss_val_log_syn_onlyfromD = []
		# loss_val_log_disc = []
		wd_log = []
		loss_train_log_syn = []
		loss_train_log_syn_onlyfromD = []
		loss_train_log_disc = []
		loss_train_smooth_log_syn = []
		loss_train_smooth_log_syn_onlyfromD = []
		loss_train_smooth_log_disc = []
	else:
		pass

	train_loader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=collate_pad_3 if train_step == 'train_text2mel' else collate_pad_2)
	validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_pad_3 if train_step == 'train_text2mel' else collate_pad_2)

	# guided attention weights.
	gaw = guided_attention_mat(cfg['MAX_TEXT_LEN'], cfg['MAX_FRAME_NUM'])

	# check model's device
	print('Model G Device:', next(model.parameters()).device)
	print('Model D Device:', next(disc.parameters()).device)

	loss_iter_G = 0
	loss_iter_G_onlyfromD = 0
	loss_iter_D = 0

	while epoch < cfg['MAX_EPOCHS']:
		print('Epoch ', epoch+1)
		print('*******************')
		loader_len = len(train_loader)

		for i, sp in enumerate(train_loader):
			# Training
			start_iter = time.time()
			opt_syn.zero_grad()
			opt_disc.zero_grad()
			
			train_target = 'D' if iteration % (cfg['RATIO']+1) else 'G'
			print('Iteration {}/{} for epoch {}, training {}'.format(str(i+1), str(loader_len), str(epoch+1), train_target))
			print('Global iteration ', iteration+1)
			
			if train_step == 'train_text2mel':
				mel_gt = sp['data_0'].to(device)
				text_id = sp['data_1'].to(device)
				spk_emb = sp['data_2'].to(device)
				B, C, T = mel_gt.shape

				spec_inputs = torch.cat((torch.zeros_like(mel_gt[:,:,:1]), mel_gt[:, :, :-1]), dim=-1)
				pred_mel_prob, att_mat = model(spec_inputs, text_id, spk_emb)

				if train_target == 'G':
					disc_syn = disc(pred_mel_prob[:, :, 1:9])

					loss_l1 = torch.mean(torch.abs(mel_gt-pred_mel_prob))
					loss_bin_div = torch.mean(-mel_gt*torch.log(pred_mel_prob+1e-8)-(1-mel_gt)*torch.log(1-pred_mel_prob+1e-8))
					att_aug = F.pad(att_mat, (0, cfg['MAX_FRAME_NUM']-att_mat.size()[-1], 0, cfg['MAX_TEXT_LEN']-att_mat.size()[-2]), value=-1)
					# Here gaw will broadcast along axis 0.
					loss_att = torch.sum(torch.ne(att_aug, -1).float()*att_aug*gaw) / torch.sum(torch.ne(att_aug, -1).float())
					loss_disc = torch.mean(-disc_syn)

					loss = loss_l1 + loss_bin_div + loss_att + loss_disc
					loss_iter_G += loss.item()
					loss_iter_G_onlyfromD += loss_disc.item()
					loss_train_log_syn.append(loss.item())
					loss_train_log_syn_onlyfromD.append(loss_disc.item())
					loss.backward()
					opt_syn.step()
					print('L1:{}, BD{}, ATT:{}, DISC:{}, ALL{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_att.item()), str(loss_disc.item()), str(loss.item())))

				if train_target == 'D':
					# coeff = torch.stack(8*[torch.stack(C*[torch.rand(B)],dim=1)],dim=2).to(device)
					# input_mid = coeff*mel_gt[:, :, 1:9].detach() + (1-coeff)*pred_mel_prob[:, :, 1:9].detach()
					# input_mid.requires_grad = True
					# output_mid = disc(input_mid)
					# # output_mid.backward(retain_graph=True, create_graph=True)
					# gradients = torch.autograd.grad(outputs=output_mid, inputs=input_mid,  grad_outputs=torch.ones(output_mid.size()).to(device), retain_graph=True, create_graph=True)[0]
					# loss_gp = torch.mean(cfg['LAMBDA']*(torch.norm(gradients,p=2,dim=(1,2))-1)**2)
					# # opt_disc.zero_grad()
					# loss_gp.backward() 

					input_gt = mel_gt[:, :, 1:9].detach()
					input_syn = pred_mel_prob[:, :, 1:9].detach()
					disc_gt = disc(input_gt)
					disc_syn = disc(input_syn)
					loss_D = torch.mean(disc_syn-disc_gt)
					loss_D.backward()
					opt_disc.step()
					disc.apply(clip_weights)
					# for p in disc.parameters():
					# 	p.data.clamp_(-0.01, 0.01)

					loss = loss_D.item()
					loss_iter_D += loss
					loss_train_log_disc.append(loss)
					wd_log.append(-loss)
					print('DISC:{}, WD:{}'.format(str(loss), str(-loss)))			

			if train_step == 'train_ssrn':
				mel_gt = sp['data_0'].to(device)
				lin_gt = sp['data_1'].to(device)
				B, C, T = lin_gt.shape

				pred_lin_prob = model(mel_gt)

				if train_target == 'G':
					disc_syn = disc(pred_lin_prob[:, :, 1:33])

					loss_l1 = torch.mean(torch.abs(lin_gt-pred_lin_prob))
					loss_bin_div = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))
					loss_disc = torch.mean(-disc_syn)

					loss = loss_l1 + loss_bin_div + loss_disc
					loss_iter_G += loss.item()
					loss_iter_G_onlyfromD += loss_disc.item()
					loss_train_log_syn.append(loss.item())
					loss_train_log_syn_onlyfromD.append(loss_disc.item())
					loss.backward()
					opt_syn.step()
					print('L1:{}, BD:{}, DISC:{}, ALL:{}'.format(str(loss_l1.item()), str(loss_bin_div.item()), str(loss_disc.item()), str(loss.item())))

				if train_target == 'D':
					# coeff = torch.stack(32*[torch.stack(C*[torch.rand(B)],dim=1)],dim=2).to(device)
					# input_mid = coeff*lin_gt[:, :, 1:33].detach() + (1-coeff)*pred_lin_prob[:, :, 1:33].detach()
					# input_mid.requires_grad = True
					# output_mid = disc(input_mid)
					# # output_mid.backward(retain_graph=True, create_graph=True)
					# gradients = torch.autograd.grad(outputs=output_mid, inputs=input_mid, grad_outputs=torch.ones(output_mid.size()).to(device), retain_graph=True, create_graph=True)[0]
					# loss_gp = torch.mean(cfg['LAMBDA']*(torch.norm(gradients,p=2,dim=(1,2))-1)**2)
					# # opt_disc.zero_grad()
					# loss_gp.backward()

					input_gt = lin_gt[:, :, 1:33].detach()
					input_syn = pred_lin_prob[:, :, 1:33].detach()
					disc_gt = disc(input_gt)
					disc_syn = disc(input_syn)
					loss_D = torch.mean(disc_syn-disc_gt)
					loss_D.backward()
					opt_disc.step()
					disc.apply(clip_weights)
					# for p in disc.parameters():
					# 	p.data.clamp_(-0.01, 0.01)
					
					loss = loss_D.item()
					loss_iter_D += loss
					loss_train_log_disc.append(loss)
					wd_log.append(-loss)				
					print('DISC:{}, WD:{}'.format(str(loss), str(-loss)))

			print('\n')
			if (iteration % cfg['VAL_EVERY_ITER'] == 0) and iteration>0:
				# print('No.{} VALIDATION'.format(str(iteration//cfg['VAL_EVERY_ITER'])))
				print('Generator average training loss: ', loss_iter_G/(cfg['VAL_EVERY_ITER']//(cfg['RATIO']+1)))
				print('Discriminator average training loss: ', loss_iter_D/(cfg['VAL_EVERY_ITER']//(cfg['RATIO']+1)*cfg['RATIO']))
				loss_train_smooth_log_syn.append(loss_iter_G/(cfg['VAL_EVERY_ITER']//(cfg['RATIO']+1)))
				loss_train_smooth_log_syn_onlyfromD.append(loss_iter_G_onlyfromD/(cfg['VAL_EVERY_ITER']//(cfg['RATIO']+1)))
				loss_train_smooth_log_disc.append(loss_iter_D/(cfg['VAL_EVERY_ITER']//(cfg['RATIO']+1)*cfg['RATIO']))
				loss_iter_G = 0
				loss_iter_G_onlyfromD = 0
				loss_iter_D = 0

				# model.eval()
				# disc.eval()
				# loss_val_syn, loss_val_syn_onlyfromD, loss_val_disc, loss_train_syn, loss_train_disc = validate(validate_loader, train_loader, gaw, cfg, model, disc, train_step)
				# loss_val_log_syn.append(loss_val_syn)
				# loss_val_log_syn_onlyfromD.append(loss_val_syn_onlyfromD)
				# loss_val_log_disc.append(loss_val_disc)
				# model.train()
				# disc.train()

				# ## How to decide current best model?
				# if loss_val_log_syn.index(min(loss_val_log_syn)) == len(loss_val_log_syn)-1:
				# 	print('Current Best Model!')
				# 	torch.save({'epoch': epoch+1,
				# 			    'iteration': iteration+1,
				# 			    'model_state_dict': model.module.state_dict() if is_parallel else model.state_dict(),
				# 			    'disc_state_dict': disc.module.state_dict() if is_parallel else disc.state_dict(),
				# 			    'opt_state_dict_syn:': opt_syn.state_dict(),
				# 			    'opt_state_dict_disc:': opt_disc.state_dict(),
				# 			    'loss_val_log_syn': loss_val_log_syn,
				# 			    'loss_val_log_syn_onlyfromD': loss_val_log_syn_onlyfromD,
				# 			    'loss_val_log_disc': loss_val_log_disc,
				# 			    'loss_train_log_syn': loss_train_log_syn,
				# 			    'loss_train_log_syn_onlyfromD': loss_train_log_syn_onlyfromD,
				# 			    'loss_train_log_disc': loss_train_log_disc,
				# 			    'loss_train_smooth_log_syn': loss_train_smooth_log_syn,
				# 			    'loss_train_smooth_log_syn_onlyfromD': loss_train_smooth_log_syn_onlyfromD,
				# 			    'loss_train_smooth_log_disc': loss_train_smooth_log_disc}, current_save_dir+'/{}_best_model.tar.pth'.format(train_step[6:]))

				# print('Generator validation loss of No.{} VALIDATION: {} on val set. {} on train set.'.format(str(iteration//cfg['VAL_EVERY_ITER']), str(loss_val_syn), str(loss_train_syn)))
				# print('Discriminator validation loss of No.{} VALIDATION: {} on val set. {} on train set.'.format(str(iteration//cfg['VAL_EVERY_ITER']), str(loss_val_disc), str(loss_train_disc)))

				torch.save({'epoch': epoch+1,
							'iteration': iteration+1,
							'model_state_dict': model.module.state_dict() if is_parallel else model.state_dict(),
							'disc_state_dict': disc.module.state_dict() if is_parallel else disc.state_dict(),
							'opt_state_dict_syn:': opt_syn.state_dict(),
							'opt_state_dict_disc:': opt_disc.state_dict(),
							# 'loss_val_log_syn': loss_val_log_syn,
							# 'loss_val_log_syn_onlyfromD': loss_val_log_syn_onlyfromD,
							# 'loss_val_log_disc': loss_val_log_disc,
							'wd_log': wd_log,
							'loss_train_log_syn': loss_train_log_syn,
							'loss_train_log_syn_onlyfromD': loss_train_log_syn_onlyfromD,
							'loss_train_log_disc': loss_train_log_disc,
							'loss_train_smooth_log_syn': loss_train_smooth_log_syn,
							'loss_train_smooth_log_syn_onlyfromD': loss_train_smooth_log_syn_onlyfromD,
							'loss_train_smooth_log_disc': loss_train_smooth_log_disc}, current_save_dir+'/{}_iteration_{}.tar.pth'.format(train_step[6:], str(iteration+1)))
				print('At iteration {} {} modelsaved at {}.'.format(str(iteration+1), train_step[6:], current_save_dir))

				if train_step == 'train_text2mel':
					plot_attention(att=att_mat[0, :, :], iters=iteration+1, fig_dir=fig_dir)
				
				losses = {'wd':wd_log, 't_s':loss_train_log_syn, 't_s_o':loss_train_log_syn_onlyfromD, 't_d':loss_train_log_disc, 't_s_s':loss_train_smooth_log_syn, 't_s_s_o':loss_train_smooth_log_syn_onlyfromD, 't_s_d':loss_train_smooth_log_disc}
				plot_loss(losses, iteration+1, fig_dir)

			end_iter = time.time()
			iteration += 1
			print('Time elapsed {}s.'.format(str(end_iter-start_iter)))

		epoch += 1
