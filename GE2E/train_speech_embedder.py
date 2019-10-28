#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import argparse
import random
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

def plot_attention(att, E, B, N, fig_dir):
    if torch.cuda.is_available():
        att = att.cpu()
    att = att.detach().numpy()

    if not os.path.exists(fig_dir):
        os.system('mkdir -p '+fig_dir)

    fig, ax = plt.subplots()
    img = ax.imshow(att)

    fig.colorbar(img)
    plt.title('Similarity Matrix')
    plt.xticks(np.arange(0,N,1))
    plt.yticks(np.arange(0,N,1))
    plt.savefig(fig_dir+'sm_test-epoch{}_batch{}.png'.format(str(E+1), str(B+1)), format='png')
    plt.close(fig)

def train(model_path):
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed(shuffle=True)
    else:
        train_dataset = SpeakerDatasetTIMIT()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    
    embedder_net = SpeechEmbedder().to(device)
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
    ge2e_loss = GE2ELoss(device)
    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader): 
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

def test(model_path,enroll_num):
    
    with torch.no_grad():
        if hp.data.data_preprocessed:
            test_dataset = SpeakerDatasetTIMITPreprocessed()
        else:
            test_dataset = SpeakerDatasetTIMIT()
        test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
        
        embedder_net = SpeechEmbedder()
        embedder_net.load_state_dict(torch.load(model_path))
        embedder_net.eval()
        
        avg_EER = 0
        avg_spoofrate = 0
        for e in range(hp.test.epochs):
            batch_avg_EER = 0
            batch_avg_spoofrate = 0
            for batch_id, mel_db_batch in enumerate(test_loader):
                assert hp.test.M % 2 == 0
                # enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
                size_1 = mel_db_batch.shape[1]
                es1 = 2*enroll_num
                enrollment_batch = mel_db_batch[:, :es1, :, :]
                verification_batch = mel_db_batch[:, es1:, :, :]
                # print(mel_db_batch.shape)
                
                enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*es1, enrollment_batch.size(2), enrollment_batch.size(3)))
                verification_batch = torch.reshape(verification_batch, (hp.test.N*(size_1-es1), verification_batch.size(2), verification_batch.size(3)))
                # print(enrollment_batch.shape)
                # print(verification_batch.shape)
                
                perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i,j in enumerate(perm):
                    unperm[j] = i
                    
                # verification_batch = verification_batch[perm]
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)
                # verification_embeddings = verification_embeddings[unperm]
                # print(enrollment_embeddings.shape)
                # print(verification_embeddings.shape)
                
                enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, enrollment_batch.size(0)//hp.test.N, enrollment_embeddings.size(1)))
                verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, verification_batch.size(0)//hp.test.N, verification_embeddings.size(1)))
                print('\nEmbeddings shape: ', enrollment_embeddings.shape)
                
                enrollment_centroids = get_centroids(enrollment_embeddings)
                print('Centroids shape: ', enrollment_centroids.shape)
                
                sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
                print('Similarity Matrix: ', sim_matrix.shape)
                if not os.path.exists(hp.save_simmat_dir):
                    os.system('mkdir -p '+hp.save_simmat_dir)
                torch.save(sim_matrix, hp.save_simmat_dir+'/simmat_e{}_b{}'.format(str(e+1), str(batch_id+1)))

                fig_dir = './simmat/sub_32/'
                plot_attention(torch.mean(sim_matrix, dim=1), e, batch_id, hp.test.N, fig_dir)
                
                # calculating EER
                diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
                
                for thres in [0.01*i+0.5 for i in range(50)]:
                    # print('Threshold: ', thres)
                    sim_matrix_thresh = sim_matrix>thres
                    FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])/(hp.test.N-1.0)/(float(size_1-es1))/hp.test.N)
                    FRR = (sum([size_1-es1-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])/(float(size_1-es1))/hp.test.N)
                    # print('FAR:{}, FRR:{}'.format(str(FAR), str(FRR)))
                    # Save threshold when FAR = FRR (=EER)
                    # spoof_simmat = sim_matrix_thresh[i,-(size_1-es1)//2:,i]
                    gtfrr = (sum([size_1//2-es1//2-sim_matrix_thresh[i,:(size_1-es1)//2,i].float().sum() for i in range(int(hp.test.N))]) / (float(size_1/2-es1/2))/hp.test.N)
                    spoof_rate = (sum([sim_matrix_thresh[i,-(size_1-es1)//2:,i].float().sum() for i in range(int(hp.test.N))]) / (float(size_1/2-es1/2))/hp.test.N)

                    if diff> abs(FAR-FRR):
                        diff = abs(FAR-FRR)
                        EER = (FAR+FRR)/2
                        EER_thresh = thres
                        EER_FAR = FAR 
                        EER_FRR = FRR
                        gt_FRR = gtfrr
                        SPOOF_RATE = spoof_rate
                batch_avg_EER += EER
                batch_avg_spoofrate += SPOOF_RATE
                print("\nEER : %0.4f (thres:%0.4f)"%(EER,EER_thresh))
            avg_EER += batch_avg_EER/(batch_id+1)
            avg_spoofrate += batch_avg_spoofrate/(batch_id+1)
        avg_EER = avg_EER / hp.test.epochs
        avg_spoofrate = avg_spoofrate / hp.test.epochs
        print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        print("\n Spoof rate across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_spoofrate))

def test_nospoof(model_path,enroll_num,eval_num):  
    with torch.no_grad():
        if hp.data.data_preprocessed:
            test_dataset = SpeakerDatasetTIMITPreprocessed()
        else:
            test_dataset = SpeakerDatasetTIMIT()
        test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
        
        embedder_net = SpeechEmbedder()
        embedder_net.load_state_dict(torch.load(model_path))
        embedder_net.eval()
        
        avg_EER = 0
        avg_thres = 0
        # avg_spoofrate = 0
        for e in range(hp.test.epochs):
            batch_avg_EER = 0
            batch_avg_thres = 0
            # batch_avg_spoofrate = 0
            for batch_id, mel_db_batch in enumerate(test_loader):
                assert hp.test.M % 2 == 0
                # enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
                size_1 = mel_db_batch.shape[1]
                es1 = 2*enroll_num
                enrollment_batch = mel_db_batch[:, :es1, :, :]
                verification_batch = mel_db_batch[:, es1:, :, :]
                # print(mel_db_batch.shape)
                
                enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*es1, enrollment_batch.size(2), enrollment_batch.size(3)))
                verification_batch = torch.reshape(verification_batch, (hp.test.N*(size_1-es1), verification_batch.size(2), verification_batch.size(3)))
                # print(enrollment_batch.shape)
                # print(verification_batch.shape)
                
                perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i,j in enumerate(perm):
                    unperm[j] = i
                    
                # verification_batch = verification_batch[perm]
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)
                # verification_embeddings = verification_embeddings[unperm]
                # print(enrollment_embeddings.shape)
                # print(verification_embeddings.shape)
                
                enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, enrollment_batch.size(0)//hp.test.N, enrollment_embeddings.size(1)))
                verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, verification_batch.size(0)//hp.test.N, verification_embeddings.size(1)))
                print('\nEmbeddings shape: ', enrollment_embeddings.shape)
                
                enrollment_centroids = get_centroids(enrollment_embeddings)
                print('Centroids shape: ', enrollment_centroids.shape)
                
                sim_matrix = get_cossim(verification_embeddings[:,:2*eval_num,:], enrollment_centroids)
                print('Similarity Matrix: ', sim_matrix.shape)

                # fig_dir = './simmat/sub_23/'
                # plot_attention(torch.mean(sim_matrix, dim=1), e, batch_id, hp.test.N, fig_dir)
                
                # calculating EER
                diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
                
                for thres in [0.01*i+0.5 for i in range(50)]:
                    # print('Threshold: ', thres)
                    sim_matrix_thresh = sim_matrix>thres
                    FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])/(hp.test.N-1.0)/(float(size_1/2-es1/2))/hp.test.N)
                    FRR = (sum([size_1//2-es1//2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])/(float(size_1/2-es1/2))/hp.test.N)
                    # print('FAR:{}, FRR:{}'.format(str(FAR), str(FRR)))
                    # Save threshold when FAR = FRR (=EER)
                    # spoof_simmat = sim_matrix_thresh[i,-(size_1-es1)//2:,i]
                    # gtfrr = (sum([size_1//2-es1//2-sim_matrix_thresh[i,:(size_1-es1)//2,i].float().sum() for i in range(int(hp.test.N))]) / (float(size_1/2-es1/2))/hp.test.N)
                    # spoof_rate = (sum([sim_matrix_thresh[i,-(size_1-es1)//2:,i].float().sum() for i in range(int(hp.test.N))]) / (float(size_1/2-es1/2))/hp.test.N)

                    if diff> abs(FAR-FRR):
                        diff = abs(FAR-FRR)
                        EER = (FAR+FRR)/2
                        EER_thresh = thres
                        EER_FAR = FAR 
                        EER_FRR = FRR
                        # gt_FRR = gtfrr
                        # SPOOF_RATE = spoof_rate
                batch_avg_EER += EER
                batch_avg_thres += EER_thresh
                # batch_avg_spoofrate += SPOOF_RATE
                print("\nEER : %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
            avg_EER += batch_avg_EER/(batch_id+1)
            avg_thres += batch_avg_thres/(batch_id+1)
            # avg_spoofrate += batch_avg_spoofrate/(batch_id+1)
        avg_EER = avg_EER / hp.test.epochs
        avg_thres = avg_thres / hp.test.epochs
        # avg_spoofrate = avg_spoofrate / hp.test.epochs
        # print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        print("\n Average threshold: ", avg_thres)
        # print("\n Spoof rate across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_spoofrate))
    return avg_thres
            
if __name__=="__main__":
    ps = argparse.ArgumentParser()
    # ps.add_argument('--train_spk_num', type=int, default=88)
    ps.add_argument('--enroll_num', type=int, default=3)
    ps.add_argument('--eval_num', type=int, default=20)
    args = ps.parse_args()
    if hp.training:
        train(hp.model.model_path)
    else:
        print('***********Mixture***********')
        test(hp.model.model_path,args.enroll_num)
        print('***********No Spoof***********')
        EER_thres = test_nospoof(hp.model.model_path,args.enroll_num,args.eval_num)
        simmat_list = os.listdir(hp.save_simmat_dir)
        spoof_rate = 0
        for k in simmat_list:
            mat = torch.load(hp.save_simmat_dir+'/'+k)
            mat_thres = mat > EER_thres
            tp = (sum([mat_thres[i,-2*args.eval_num:,i].float().sum() for i in range(int(hp.test.N))]) / (float(2*args.eval_num))/hp.test.N)
            spoof_rate += tp.item()
            print('\n', k, 'Spoof Rate: ', tp.item())
        spoof_rate = spoof_rate / len(simmat_list)
        print('\nAverage spoof rate: ', spoof_rate)
