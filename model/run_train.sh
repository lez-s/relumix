CUDA_VISIBLE_DEVICES=1 python train_relit.py \
    --base=configs/train_relumix.yaml \
    --name=relumix --seed=1234 --num_nodes=1 --wandb=0 \
