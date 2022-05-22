

GPU_ID=0 #$1
DATASET='rsvg' #$2

IMDB="dota_v1_0"
TAG="RoITransformer"
NET="res50"
ID="mrcn_cmr_with_st"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --max_iters 30000 \
    --with_st 1 \
    --id ${ID}
