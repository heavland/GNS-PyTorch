# GNS-PyTorch
A PyTorch implementation of the “Graph Network-based Simulators” (GNS) model ([Learning to simulate complex physics with graph networks](https://arxiv.org/abs/2002.09405), ICML 2020) from DeepMind for simulating particle-based dynamics using graph networks. 

This repo uses purely PyTorch's native APIs, and is a re-implementation of the [official one](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) (which is wrapped by the infamous TF Estimator and relies on several external libraries, making it impossible to play with).

### Environment
Set up conda env and install dependencies:
```
conda env create -f environment.yml
conda activate GNS-PyTorch
```

This environment is upgraded for modern GPUs (including RTX 50 series) with PyTorch + CUDA 12.1.

Verify installation:
```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

If your driver/CUDA combination differs, regenerate install command from:
https://pytorch.org/get-started/locally/

### Data
Download datasets and convert to pytorch-friendly format:
```
bash download_dataset.sh WaterRamps tfdatasets/
python extract_tfrs.py --data-path tfdatasets/WaterRamps
rm -rf tfdatasets/WaterRamps
```

### Training
Training with tensorboard logging and model saving:
```
python train.py --cfg configs/dmwater.yaml --exp-name dmwater
```
Tensorboard log files are saved under `logs/{exp-name}`.

### Evaluation
Evaluate using saved checkpoint:
```
python eval.py --cfg configs/dmwater.yaml --ckpt ckpts/dmwater/iter_25000.path.tar --data-dir WaterRamps/test
```
Rollout visualization gifs are saved under `eval_vis/`.

### Result
Example results after 2 million training steps:

![0](gifs/0.gif)
![1](gifs/1.gif)
![2](gifs/2.gif)
