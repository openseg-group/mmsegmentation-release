#!/usr/bin/env bash
PYTHON="/data/anaconda/envs/pytorch1.5.1/bin/python"
$PYTHON -m pip install -e .

# 81.03
# CONFIG="configs/ocrnet/ocrnetplus_r101-d8_512x1024_60k_b16_cityscapes.py"
# CHECKPOINT="/home/yuhui/teamdrive/yuyua/code/segmentation/mmsegmentation/work_dirs/ocrnetplusv2_r101-d8_bs2x_sep_512x1024_60k_cityscapes_run3/iter_60000.pth"
# GPUS=4
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# ${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU cityscapes

# 80.29
# CONFIG="configs/ocrnet/ocrnet_r101-d8_512x1024_40k_b16_cityscapes.py"
# CHECKPOINT="/home/yuhui/teamdrive/yuyua/code/segmentation/mmsegmentation/work_dirs/ocrnet_r101-d8_bs2x_512x1024_40k_cityscapes/iter_40000.pth"
# GPUS=4
# PORT=${PORT:-29500}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# ${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU cityscapes


# 79.88
CONFIG="configs/ocrnet/ocrnet_r101-d8_512x1024_40k_cityscapes.py"
CHECKPOINT="/home/yuhui/teamdrive/yuyua/code/segmentation/mmsegmentation/work_dirs/ocrnet_r101-d8_512x1024_40k_cityscapes/iter_40000.pth"
GPUS=4
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU cityscapes
