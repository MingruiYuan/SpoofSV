import os
import argparse
from train.ordinary import ordinary_train
from train.adversarial_wasserstein_gp import adversarial_train
from synthesize import synthesize
import json

if __name__ == '__main__':
    ps = argparse.ArgumentParser(description='Adversarial Conditional Text-to-speech')
    ps.add_argument('step', choices=['train_text2mel', 'train_ssrn', 'synthesize'], metavar='s')
    ps.add_argument('-P','--pattern', choices=['universal', 'conditional', 'ubm-finetune'], default='conditional', metavar='m')
    ps.add_argument('-R','--resume', type=str, default=None, metavar='checkpoint')
    ps.add_argument('-C', '--configuration', type=str, default=None)
    ps.add_argument('--adversarial', action='store_true')
    ps.add_argument('--save_spectrogram', action='store_true')
    ps.add_argument('-T','--current_time', type=str, required=True, metavar='T')
    args = ps.parse_args()

    with open(args.configuration, 'r') as f:
        config = json.load(f)

    if args.save_spectrogram:
        spec_dir = config['SRC_ROOT_DIR'] + 'spec/'
        if (not os.path.exists(spec_dir)):
            os.system('mkdir -p '+spec_dir)
    else:
        spec_dir = None

    if args.step in ['train_text2mel', 'train_ssrn']:
        if args.adversarial:
            adversarial_train(train_step=args.step,
                              train_pattern=args.pattern,
                              cfg=config,
                              spec_dir=spec_dir,
                              resume_checkpoints=args.resume,
                              current_time=args.current_time)
        else:
            ordinary_train(train_step=args.step,
                           train_pattern=args.pattern,
                           cfg=config,
                           spec_dir=spec_dir,
                           resume_checkpoints=args.resume,
                           current_time=args.current_time)

    if args.step == 'synthesize':
        synthesize(pattern=args.pattern,
                   cfg=config,
                   spec_dir=spec_dir,
                   current_time=args.current_time)