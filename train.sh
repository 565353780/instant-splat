SOURCE_PATH=$HOME/chLi/Dataset/GS/haizei_1/input/
MODEL_PATH=$HOME/chLi/Dataset/GS/haizei_1/instant-splat/
N_VIEW=98
GPU_ID=3,4,5

# (1) Co-visible Global Geometry Initialization
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  --n_views ${N_VIEW} \
  --focal_avg \
  --co_vis_dsp \
  --conf_aware_ranking \
  --infer_video \
  >${MODEL_PATH}/01_init_geo.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"

# (2) Train: jointly optimize pose
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEW} \
  --iterations ${gs_train_iter} \
  --pp_optimizer \
  --optim_pose \
  >${MODEL_PATH}/02_train.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"

# (3) Render-Video
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEW} \
  --iterations ${gs_train_iter} \
  --infer_video \
  >${MODEL_PATH}/03_render.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render.log"

echo "======================================================="
echo "Task completed: (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
echo "======================================================="
