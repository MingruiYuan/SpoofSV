import os
import argparse

ps = argparse.ArgumentParser()
ps.add_argument('-S', '--score_file', type=str, default=None)
ps.add_argument('--thres', type=float, default=0)
ps.add_argument('--train_spk_num', type=int, default=88)
ps.add_argument('--enroll_utt_num', type=int, default=3)
ps.add_argument('--eval_utt_num', type=int, default=20)
args = ps.parse_args()

with open(args.score_file, 'r') as f:
	scores = f.readlines()

total_num = (len(scores)/2) // (108-args.train_spk_num)
assert total_num == (108-args.train_spk_num)*args.eval_utt_num

spoof_num = 0
for k in range(len(scores)):
	score = scores[k].strip().split()
	if (score[0] == score[1][:3]) and (int(score[1][-3:]) > args.enroll_utt_num + args.eval_utt_num):
		spoof_num += (float(score[2]) > args.thres)

print('Spoof Rate: {}'.format(str(spoof_num/total_num)))