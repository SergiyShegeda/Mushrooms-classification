export TF_CPP_MIN_LOG_LEVEL=0

alias train='python train.py mushrooms 1'
alias tensorboard='tensorboard --logdir=runs/$(date "+%Y-%m-%d")'
interference() {
    python inference.py "$1" "$2"
}