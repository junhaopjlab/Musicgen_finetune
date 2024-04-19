

export DATA="/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiSongs"  #"/mnt/petrelfs/share_data/liuzihan/acc_dataset_zh" #"/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiEN_test" #"path of data"  
export MODEL="/mnt/petrelfs/liuzihan/.cache/huggingface/hub/models--facebook--musicgen-melody-large/snapshots/6fdf8d3d815995108c9bdb5183414ff464b171ac"  #"model name or path"
export SAVE_PATH="/mnt/petrelfs/liuzihan/Musicgen_finetune/saved_models/melody_zh_cond"

#8
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
    --save_path ${SAVE_PATH} \
    --model_id ${MODEL} \
    --epochs 5 \
    --batch_size 64 \
    --save_step 500 \
    --use_wandb  1 \
    --warmup_steps 200 \
    --lr 1e-5

#debug
# torchrun $DISTRIBUTED_ARGS dist_finetune.py \
#     --dataset_path ${DATA} \
#     --save_path ${SAVE_PATH} \
#     --model_id ${MODEL} \
#     --epochs 1 \
#     --batch_size 4 \
#     --save_step 4 \
#     --use_wandb  1 \
#     --warmup_steps 2