SOURCE_PATH=$HOME/chLi/Dataset/GS/haizei_1/input_sparse/
MODEL_PATH=$HOME/chLi/Dataset/GS/haizei_1/instant-splat_sparse/
N_VIEW=5
GPU_ID=7

gs_train_iter=1000

mkdir -p $MODEL_PATH

# (1) Co-visible Global Geometry Initialization
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  --n_views ${N_VIEW} \
  --focal_avg \
  --co_vis_dsp \
  --conf_aware_ranking \
  --infer_video
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed."

# (2) Train: jointly optimize pose
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEW} \
  --iterations ${gs_train_iter} \
  --pp_optimizer \
  --optim_pose
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed."

# (3) Render-Video
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEW} \
  --iterations ${gs_train_iter} \
  --infer_video
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed."

echo "======================================================="
echo "Task completed: (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
echo "======================================================="
