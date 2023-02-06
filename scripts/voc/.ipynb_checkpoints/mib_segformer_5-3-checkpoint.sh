#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=2,3
NB_GPU=2


DATA_ROOT=../data/PascalVOC12

DATASET=voc
TASK=5-3
NAME=MiB
METHOD=MiB
OPTIONS="--checkpoint checkpoints/step/"

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/path/to/my/first/weights
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

BATCH_SIZE=12
INITIAL_EPOCHS=30
EPOCHS=30

# CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --model 'segformer_b2'
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --model 'segformer_b2' 
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --model 'segformer_b2' 
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --model 'segformer_b2'
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --model 'segformer_b2' 

# rm checkpoints/step/5-3-voc_PLOP_1_segformer_b2.pth
# rm checkpoints/step/5-3-voc_PLOP_2_segformer_b2.pth
# rm checkpoints/step/5-3-voc_PLOP_3_segformer_b2.pth
# rm checkpoints/step/5-3-voc_PLOP_4_segformer_b2.pth

# rm checkpoints/step/15-5s-voc_MiB_1_segformer_b2.pth
# rm checkpoints/step/15-5s-voc_MiB_2_segformer_b2.pth
# rm checkpoints/step/15-5s-voc_MiB_3_segformer_b2.pth
# rm checkpoints/step/15-5s-voc_MiB_4_segformer_b2.pth

# rm checkpoints/step/5-3-voc_MiB_5_segformer_b2.pth

# rm checkpoints/step/10-1-voc_MiB_1_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_2_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_3_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_4_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_5_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_6_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_7_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_8_segformer_b2.pth
# rm checkpoints/step/10-1-voc_MiB_9_segformer_b2.pth

python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
