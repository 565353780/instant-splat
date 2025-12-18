SOURCE_PATH=$HOME/chLi/Dataset/GS/haizei_1/input_dense/
MODEL_PATH=$HOME/chLi/Dataset/GS/haizei_1/instant-splat_2dgs/
N_VIEW=98
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
  --optim_pose \
  --depth_ratio 0 \
  --lambda_dist 100 \
  --pp_optimizer
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed."

# (3) Render-Training_View
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEW} \
  --iterations ${gs_train_iter} \
  --depth_ratio 0 \
  --num_cluster 50 \
  --mesh_res 2048 \
  --depth_trunc 6.0
# --voxel_size 0.004 \
# --sdf_trunc 0.016 \
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed."
