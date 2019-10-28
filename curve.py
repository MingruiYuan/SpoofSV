import argparse
import torch
import numpy as np 
import matplotlib.pyplot as plt
from hparam import hparam as hp

ps = argparse.ArgumentParser()
ps.add_argument('--simmat', type=str, default=None)
ps.add_argument('--ivector_score', type=str, default=None)
args = ps.parse_args()

# fig = plt.subplot(331)
fig, ax = plt.subplots(1,1)

thresholds = [0.0001*i+0.5 for i in range(5000)]
spoof_rate_log = []
gt_frr_log = []
sim = torch.load(args.simmat, map_location='cpu')
for thres in thresholds:
    sim_thres = sim > thres
    spoof_rate = (sum([sim_thres[i,-2*20:,i].float().sum() for i in range(int(hp.test.N))]) / (float(2*20))/hp.test.N)
    gt_frr = (sum([2*20-sim_thres[i,:2*20,i].float().sum() for i in range(int(hp.test.N))])/(float(2*20))/hp.test.N)
    spoof_rate_log.append(spoof_rate.item())
    gt_frr_log.append(gt_frr.item())
    print('Threshold:{}, Spoof Rate:{}. GT FRR:{}\n'.format(str(thres), str(spoof_rate.item()), str(gt_frr.item())))

real_score = []
fake_score = []
spoof_rate_log_2 = []
gt_frr_log_2 = []
thresholds = [-50+0.01*i for i in range(8000)]
with open(args.ivector_score, 'r') as f:
    contents = f.readlines()
    for k, content in enumerate(contents):
        info = content.strip().split()
        if int(info[1][-3:]) > 23 and info[0]==info[1][:3]:
            fake_score.append(float(info[-1]))
        if int(info[1][-3:]) <= 23 and info[0]==info[1][:3]:
            real_score.append(float(info[-1]))
real_score = np.array(real_score)
fake_score = np.array(fake_score)
assert len(real_score) == len(fake_score)
L = len(real_score)

for thres in thresholds:
    t1 = real_score > thres
    t2 = fake_score > thres
    spoof_rate_log_2.append(np.sum(t2)/L)
    gt_frr_log_2.append(1-np.sum(t1)/L)

ax.plot(spoof_rate_log,gt_frr_log,'r--',lw=1)
ax.plot(spoof_rate_log_2,gt_frr_log_2,'b',lw=1)
ax.set_xlabel('Spoof Rate')
ax.set_ylabel('FRR in real speech')
ax.legend(['GE2E', 'i-vectors'])
plt.savefig('curve.png', format='png')