export NODE_RANK=${RANK}
echo $NODE_RANK $MASTER_ADDR $MASTER_PORT
NCCL_DEBUG=INFO
torchrun --nnodes=12 --nproc_per_node=8 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/train.py \
    configs/piflux/gmflux_2nfe_k16.py \
    --launcher pytorch --diff_seed
