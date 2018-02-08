#!/bin/sh 
#tmux new-session -s 'train' -d 'python trainer.py -b=20'
tmux new-session -s 'train' -d 'watch -n 0.1 nvidia-smi'
#tmux new-session -s 'Monitor' -d 'watch -n 0.1 nvidia-smi'
tmux split-window -v 'htop'
tmux split-window -h 'tensorboard --logdir=./runs/ --port=12345'
tmux set -g remain-on-exit on
tmux new-window 'python trainer.py -b=20'
tmux new-window 'python trainer.py -b=40'
tmux new-window 'python trainer.py -b=60'
tmux new-window 'python trainer.py -b=80'
tmux new-window 'python trainer.py -b=100'
tmux new-window 'python trainer.py -b=120'
#tmux select-window -t 0
tmux -2 attach-session -d 
