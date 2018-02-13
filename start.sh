tmux new -s "monitor" -d "python trainer.py"
#tmux split-window -h "python time.py" #<---- write this file
tmux split-window -v "htop"
tmux split-window -h "watch -n 0.1 nvidia-smi"
tmux set-g remain-on-exit on
tmux new-window "Tensorboard --logdir=./logs/ port=0000"
tmux -2 attach-session -d
