export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=-1
alias train='python train.py mushrooms 20'
alias tensorboard='tensorboard --logdir=runs/$(date "+%Y-%m-%d")'
inference() {
    python inference.py "$1" "$2" "$3"
}
