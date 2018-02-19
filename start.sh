tmux new-session -s "monitor" -d
tmux send-keys -t "monitor" "./hp_search.sh" C-m
tmux split-window -v "htop"
tmux split-window -h "watch -n 0.1 nvidia-smi"
tmux set -g remain-on-exit on
tmux new-window "Tensorboard --logdir=./logs/ port=12345"
tmux select-window -t 0
tmux -2 attach-session -d
#tmux split-window -h "python time.py" #<---- write this file
