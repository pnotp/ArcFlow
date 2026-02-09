torchrun --nnodes=1 --nproc_per_node=4 train.py \
    configs/qwen/arcqwen_2nfe_k16.py \
    --launcher pytorch --diff_seed