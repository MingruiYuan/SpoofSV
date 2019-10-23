import os
import numpy.random
import json
import argparse
import librosa

ps = argparse.ArgumentParser(description='Genrate meta-data')
ps.add_argument('-c', '--config_path', type=str)
args = ps.parse_args()

with open(args.config_path, 'r') as f:
    config = json.load(f)

root_dir = config["DATA_ROOT_DIR"] # Your data root directory.
os.system('export ROOT_DIR='+root_dir)
os.system('rm -rf ${ROOT_DIR}wav48/p315/')
os.system('rm -f ${ROOT_DIR}wav48/p376/p376_295.raw')
wav = root_dir + 'wav48/'
txt = root_dir + 'txt/'
new_wav_dir = root_dir + 'wav22/'

file1 = open(root_dir+'wav.path.train','w')  
file2 = open(root_dir+'txt.path.train','w')
file3 = open(root_dir+'wav.path.validate','w')
file4 = open(root_dir+'txt.path.validate','w')
file5 = open(root_dir+'wav.path.synthesize','w')
file6 = open(root_dir+'txt.path.synthesize','w')
dev_loc = [1/7,2/7,5/7]
test_loc = [3/7,4/7,6/7]
for k in range(len(os.listdir(wav))):
    wv = os.listdir(wav+os.listdir(wav)[k])
    if not os.path.exists(new_wav_dir+os.listdir(wav)[k]):
    	os.system('mkdir -p '+new_wav_dir+os.listdir(wav)[k])
    wv.sort(key = lambda x:x[:-4])
    tx = os.listdir(txt+os.listdir(txt)[k])
    tx.sort(key = lambda x:x[:-4])
    dev_loc_1 = [int(len(wv)*dev_loc[k1]) for k1 in range(len(dev_loc))]
    test_loc_1 = [int(len(tx)*test_loc[k1]) for k1 in range(len(test_loc))]
    for p in range(len(wv)):
    	print('Process ', wv[p])
    	y, sr = librosa.load(path=wav+os.listdir(wav)[k]+'/'+wv[p],mono=True)
        assert sr==22050
    	librosa.output.write_wav(path=new_wav_dir+os.listdir(wav)[k]+'/'+wv[p],y=y,sr=sr)
    	if p in dev_loc_1:
    		file3.write(new_wav_dir+os.listdir(wav)[k]+'/'+wv[p]+'\n')
    		file4.write(txt+os.listdir(txt)[k]+'/'+tx[p]+'\n')
    	elif p in test_loc_1:
    		file5.write(new_wav_dir+os.listdir(wav)[k]+'/'+wv[p]+'\n')
    		file6.write(txt+os.listdir(txt)[k]+'/'+tx[p]+'\n')
    	else:
	        file1.write(new_wav_dir+os.listdir(wav)[k]+'/'+wv[p]+'\n')
	        file2.write(txt+os.listdir(txt)[k]+'/'+tx[p]+'\n')
file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
os.system('mkdir -p ${ROOT_DIR}data_path/ordinary')
os.system('mv wav.* ${ROOT_DIR}data_path/ordinary/')
os.system('mv txt.* ${ROOT_DIR}data_path/ordinary/')