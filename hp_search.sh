#!/bin/sh 
tmux new-session -s 'train' -d 'watch -n 0.1 nvidia-smi'
tmux split-window -v 'htop'
tmux split-window -h 'tensorboard --logdir=./logs/ --port=12345'
tmux set -g remain-on-exit on
tmux new-window 'python trainer.py -b=20'
#tmux select-window -t 0
tmux -2 attach-session -d 
