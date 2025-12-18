conda install cmake -y

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install roma evo gradio matplotlib tqdm opencv-python \
  scipy einops trimesh tensorboard gdown cython icecream \
  plyfile open3d ninja imageio[ffmpeg]

pip install pyglet <2
pip install huggingface-hub[torch] >=0.22

cd submodules/simple-knn
python setup.py install

cd ../diff-gaussian-rasterization
python setup.py install

cd ../fused-ssim
python setup.py install

cd ../../croco/models/curope/
python setup.py build_ext --inplace
