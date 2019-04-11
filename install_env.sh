conda create -n east python=3.6 tensorflow-gpu=1.5.0 keras=2.1.4 numpy=1.14.1
source activate east
conda install cudnn=7.0.5
pip install tqdm==4.19.7
pip install opencv-python

