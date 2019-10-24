import os
import librosa
import numpy as np
from scipy import signal
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from models.TTSModel_layer_norm import melSyn, SSRN
from data.dataset import dataset, collate_pad_4
import math
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def guided_attention_mat(max_text_len, max_frame_num):
    g = 0.2
    W = torch.zeros((max_text_len, max_frame_num))
    for k1 in range(max_text_len):
        for k2 in range(max_frame_num):
            W[k1, k2] = 1-math.exp(-(k2/max_frame_num-k1/max_text_len)**2/(2*g*g))
    W = W.to(device)
    return W

def plot_attention(att, idx, fig_dir):
    if torch.cuda.is_available():
        att = att.cpu()
    att = att.numpy()

    if not os.path.exists(fig_dir):
        os.system('mkdir -p '+fig_dir)

    fig, ax = plt.subplots()
    img = ax.imshow(att)

    fig.colorbar(img)
    plt.title('Sample from batch {}'.format(str(idx)))
    plt.savefig(fig_dir+'att_batch_{}.png'.format(str(idx)), format='png')
    plt.close(fig)

def synthesize(pattern, cfg, spec_dir, current_time=None):
    """
    Args:
    --pattern: 'universal' or 'conditional'.
    --cfg: configuration file.
    --spec_dir: None or Directory of saved spectrograms.
    --model_text2mel: Trained model of text2mel Network.
    --model_ssrn: Trained model of ssrn Network.
    """

    sample_dir = cfg['SRC_ROOT_DIR'] + 'samples/' + current_time + '/'
    fig_dir = sample_dir + 'fig/'

    if not os.path.exists(fig_dir):
        os.system('mkdir -p '+fig_dir)

    if cfg['APPLY_DROPOUT']:
        from models.TTSModel_dropout import melSyn, SSRN
    else:
        from models.TTSModel import melSyn, SSRN

    synthesize_dataset = dataset(cfg=cfg, mode='synthesize', pattern=pattern, step='synthesize', spec_dir=spec_dir)

    m1 = melSyn(vocab_len=len(cfg['VOCABULARY'])-1,
                condition = (pattern == 'conditional'),
                spkemb_dim=cfg['SPK_EMB_DIM'],
                textemb_dim=cfg['TEXT_EMB_DIM'],
                freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
                hidden_dim=cfg['HIDDEN_DIM'])

    m2 = SSRN(freq_bins=cfg['COARSE_MELSPEC']['FREQ_BINS'],
              output_bins=(1+cfg['STFT']['FFT_LENGTH']//2),
              ssrn_dim=cfg['SSRN_DIM'])

    if cfg['MULTI_GPU']:
        m1 = torch.nn.DataParallel(m1)
        m2 = torch.nn.DataParallel(m2)

    print('CUDA available: ', torch.cuda.is_available())
    ckp1 = torch.load(cfg['INFERENCE_TEXT2MEL_MODEL'])
    ckp2 = torch.load(cfg['INFERENCE_SSRN_MODEL'])
    m1.load_state_dict(ckp1['model_state_dict'])
    m2.load_state_dict(ckp2['model_state_dict'])
    m1.to(device)
    m1.eval()    
    m2.to(device)
    m2.eval()

    synthesize_loader = DataLoader(synthesize_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_pad_4)

    gaw = guided_attention_mat(cfg['MAX_TEXT_LEN'], cfg['MAX_FRAME_NUM'])

    loss_avg_t2m = 0
    loss_avg_ssrn = 0

    with torch.no_grad():
        for i, sp in enumerate(synthesize_loader):
            mel_gt = sp['data_0'].to(device)
            text_id = sp['data_1'].to(device)
            spk_emb = sp['data_2'].to(device)
            lin_gt = sp['data_3'].to(device)

            d1, d2, d3 = mel_gt.shape
            init_frame = torch.zeros((d1, d2, 1)).to(device)
            Y, A, prev_maxatt, K, V = m1(melspec=init_frame, textid=text_id, spkemb=spk_emb, pma=torch.zeros((d1,)).long().to(device))
            inputs = torch.cat((init_frame, Y), dim=-1)
            for frame in range(d3-1):
                Y, A, prev_maxatt = m1(melspec=inputs, textid=None, spkemb=spk_emb, K=K, V=V, A_last=A, pma=prev_maxatt)
                inputs = torch.cat((inputs, Y[:, :, -1:]), dim=-1)

            plot_attention(att=A[0, :, :], idx=i+1, fig_dir=fig_dir)

            loss_l1_t2m = torch.mean(torch.abs(mel_gt-Y))
            loss_bin_div_t2m = torch.mean(-mel_gt*torch.log(Y+1e-8)-(1-mel_gt)*torch.log(1-Y+1e-8))
            A_aug = F.pad(A, (0, cfg['MAX_FRAME_NUM']-A.size()[-1], 0, cfg['MAX_TEXT_LEN']-A.size()[-2]), value=-1)
            loss_att = torch.sum(torch.ne(A_aug, -1).float()*A_aug*gaw) / torch.sum(torch.ne(A_aug, -1).float())

            loss_t2m = loss_l1_t2m + loss_bin_div_t2m + loss_att
            loss_avg_t2m += loss_t2m.item()
            print('syn set text2mel loss: {} {} {} {}'.format(str(loss_l1_t2m.item()), str(loss_bin_div_t2m.item()), str(loss_att.item()), str(loss_t2m.item())))

            pred_lin_prob = m2(Y)
            loss_l1_ssrn = torch.mean(torch.abs(lin_gt-pred_lin_prob))
            loss_bin_div_ssrn = torch.mean(-lin_gt*torch.log(pred_lin_prob+1e-8)-(1-lin_gt)*torch.log(1-pred_lin_prob+1e-8))

            loss_ssrn = loss_l1_ssrn + loss_bin_div_ssrn
            loss_avg_ssrn += loss_ssrn.item()
            print('syn set ssrn loss: {} {} {}'.format(str(loss_l1_ssrn.item()), str(loss_bin_div_ssrn.item()), str(loss_ssrn.item())))

            if torch.cuda.is_available():
                pred_lin_prob = pred_lin_prob.cpu()
            pred_lin = pred_lin_prob.numpy()

            if cfg['LOG_FEATURE']:
                pred_lin = pred_lin*cfg['MAX_DB'] - cfg['MAX_DB'] + cfg['REF_DB']
                pred_lin = np.power(10, 0.05*pred_lin)
    
            for k in range(d1):
                # print(pred_lin[k, :, :])
                # print(np.max(np.max(pred_lin[k, :, :])))
                if not cfg['LOG_FEATURE']:
                    pred_lin[k, :, :] = pred_lin[k, :, :]/np.max(pred_lin[k, :, :])
                spec = pred_lin[k, :, :]**(cfg['NORM_POWER']['RECONSTRUCTION']/cfg['NORM_POWER']['ANALYSIS'])
                time_signal = librosa.core.griffinlim(S=spec, n_iter=64, hop_length=cfg['STFT']['HOP_LENGTH'], win_length=cfg['STFT']['FFT_LENGTH'])
                time_signal = signal.lfilter([1], [1, -cfg['PREEMPH']], time_signal)
                #print(time_signal, np.max(time_signal))
                librosa.output.write_wav(sample_dir+'S{}_B{}.wav'.format(str(k+1), str(i+1)), time_signal if cfg['LOG_FEATURE'] else time_signal/np.max(time_signal)*0.75, cfg['SAMPLING_RATE'])
