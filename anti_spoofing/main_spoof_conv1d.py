import os
import time
import json
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from discriminator import melDisc, linDisc, melDisc_v1, linDisc_v1, melDisc_v2, linDisc_v2
from spoof_conv1d import ASVspoofDataset, collate_pad_3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ps = argparse.ArgumentParser()
    ps.add_argument('step', choices=['train', 'dev'], metavar='s')
    ps.add_argument('-T', '--time', type=str, required=True)
    ps.add_argument('-R', '--resume', type=str, default=None)
    ps.add_argument('-C', '--configuration', type=str, required=True)
    ps.add_argument('--variant', type=str)
    args = ps.parse_args()

    with open(args.configuration, 'r') as f:
        cfg = json.load(f)

    step = args.step
    save_dir = './checkpoints/' + args.time
    feat_type = 'mel'
    save_interval = 1000
    max_epochs = 20000
    resume_checkpoints = args.resume

    if step == 'train':
        model = melDisc(cfg['COARSE_MELSPEC']['FREQ_BINS'], cfg['DISC_DIM']) if feat_type == 'mel' else linDisc((1+cfg['STFT']['FFT_LENGTH']//2), cfg['DISC_DIM'])
    else:
        if args.variant == 'v1':
            model = melDisc_v1(cfg['COARSE_MELSPEC']['FREQ_BINS'], cfg['DISC_DIM']) if feat_type == 'mel' else linDisc_v1((1+cfg['STFT']['FFT_LENGTH']//2), cfg['DISC_DIM'])
        if args.variant == 'v2':
            model = melDisc_v2(cfg['COARSE_MELSPEC']['FREQ_BINS'], cfg['DISC_DIM']) if feat_type == 'mel' else linDisc_v2((1+cfg['STFT']['FFT_LENGTH']//2), cfg['DISC_DIM'])

    if (not os.path.exists(save_dir)) and step == 'train':
        os.system('mkdir -p ' + save_dir)

    if not os.path.exists('./cm_scores'):
        os.system('mkdir cm_scores')
    
    if resume_checkpoints is None:
        epoch = 0
        global_iteration = 0
        # loss_val_log = []
        model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
    else:
        ckpt = torch.load(resume_checkpoints)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        global_iteration = ckpt['global_iteration']
        # loss_val_log = ckpt['loss_val_log']

    corpus = ASVspoofDataset(cfg=cfg, step=step, time=args.time)
    loader = DataLoader(corpus, 64, shuffle=(step=='train'), num_workers=4, collate_fn=collate_pad_3)

    print('Model device:', next(model.parameters()).device)

    if step == 'train':
        while epoch < max_epochs:
            print('Epoch ', epoch+1)
            print('*******************')

            for i, sp in enumerate(loader):
                mel = sp['data_0']
                lin = sp['data_1']
                label = sp['data_2']

                start_iter = time.time()
                optimizer.zero_grad()
                feat = mel.to(device) if feat_type == 'mel' else lin.to(device)
                label = torch.squeeze(label).to(device)

                pred = model(feat).squeeze() 
                # print(feat.shape)
                # print(pred.shape)
                # print(label.shape)
                loss = (-label*torch.log(pred+1e-6)-(1-label)*torch.log(1-pred+1e-6)).mean()
                # print(loss.shape)
                loss.backward()
                optimizer.step()

                end_iter = time.time()
                print('Epoch {}: Iteration{}/{}'.format(str(epoch+1), str(i+1), str(len(loader))))
                print('Loss: ', loss.item())
                print('Global iteration: ', global_iteration+1)
                print('Time elapsed: {}\n'.format(str(end_iter-start_iter)))

                if global_iteration % save_interval == 0 and global_iteration > 0:
                    torch.save({'epoch': epoch+1, 
                                'global_iteration': global_iteration, 
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, save_dir+'/{}_iteration.tar.pth'.format(str(global_iteration+1)))

                global_iteration += 1

            epoch += 1

    if step == 'dev':
        cm_scores = open('./cm_scores/scores_{}.txt'.format(args.time), 'w')
        model.eval()
        idx = 0
        with torch.no_grad():
            for i, sp in enumerate(loader):
                mel = sp['data_0']
                lin = sp['data_1']
                label = sp['data_2']

                print('Utterance: {}/{}'.format(str(i+1), len(loader)))
                feat = mel.to(device) if feat_type == 'mel' else lin.to(device)
                label = label.to(device)

                pred = model(feat).squeeze()
                # score = str(pred[0, 1].item() - pred[0, 0].item())
                # is_norm = torch.exp(pred[0,1]) + torch.exp(pred[0,0])
                # print('b_prob:{} + s_prob:{} = {}'.format(str(torch.exp(pred[0,1]).item()), str(torch.exp(pred[0,0]).item()), str(is_norm.item())))
                for k in range(mel.shape[0]):
                    gt = 'bonafide' if label[k].item() == 1 else 'spoof'
                    print('Score: {}. Label: {}\n'.format(pred[k].item(), gt))
                    cm_scores.write('LA_D_{} - {} {}\n'.format(str(idx).zfill(7), gt, str(pred[k].item())))
                    idx += 1

        cm_scores.close()

if __name__ == '__main__':
    main()