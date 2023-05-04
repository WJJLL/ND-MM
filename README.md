# ND-MM (AAAI-2023 Oral)
Exploring Non-Target Knowledge for Improving Ensemble Universal Adversarial Attacks 
## Setup
python=3.8.12
torch=1.4.0
torchvision=0.5.0
GPU = GeForce RTX 3090
## Config
Edit the paths accordingly in `config.py`
## DataSets
The code supports training UAPs on ImageNet or other dataset
### ImageNet
The [ImageNet](http://www.image-net.org/) dataset should be preprocessed, such that the validation images are located in labeled subfolders as for the training set. You can have a look at this [bash-script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) if you did not process your data already. Set the paths in your `config.py`.
```
IMAGENET_Train_PATH = "./Data/ImageNet/train"
IMAGENET_Test_PATH = './Data/ImageNet/val'
```
* You can try your own data set as training data instead of the ImageNet train set to craft UAPs.
## Run
Run `bash run.sh` to generate UAPs or test performance. The bash script should be easy to adapt to perform different experiments.



## Defense Models
Evaluation against robust training [SIN](https://github.com/rgeirhos/Stylized-ImageNet), [Augmix](https://github.com/google-research/augmix),[Adversarial](https://github.com/microsoft/robust-models-transfer),[NPR](https://github.com/Muzammal-Naseer/NRP)
