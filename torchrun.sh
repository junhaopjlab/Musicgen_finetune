

export DATA="/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiEN_test" #"path of data"  
export MODEL="/mnt/petrelfs/liuzihan/.cache/huggingface/hub/models--facebook--musicgen-melody-large/snapshots/6fdf8d3d815995108c9bdb5183414ff464b171ac"  #"model name or path"


GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29501

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS dist_finetune.py \
    --dataset_path ${DATA} \
    --model_id ${MODEL} \
    --epochs 2