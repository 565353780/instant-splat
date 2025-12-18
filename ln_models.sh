LOCAL_MODEL_PATH=/nvme0pnt/lichanghao/chLi/Model
LOCAL_GIT_PATH=/nvme0pnt/lichanghao/github/RECON/instant-splat

mkdir -p ${LOCAL_GIT_PATH}/mast3r/checkpoints/

rm ${LOCAL_GIT_PATH}/mast3r/checkpoints/*

ln -s \
  ${LOCAL_MODEL_PATH}/MASt3R/* \
  ${LOCAL_GIT_PATH}/mast3r/checkpoints/
