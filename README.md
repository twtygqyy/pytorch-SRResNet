# PyTorch SRResNet
Implementation of Paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"(https://arxiv.org/abs/1609.04802) in PyTorch

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]
               
optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=500
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --clip CLIP           Clipping Gradients. Default=0.1
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 0
  --pretrained PRETRAINED
                        path to pretrained model (default: none)

```


### Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]

PyTorch SRResNet Test

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --image IMAGE  image name
  --scale SCALE  scale factor, Default: 4
```
We convert Set5 test set images to mat format using Matlab, for best PSNR performance, please use Matlab
An example of usage is shown as follows:
```
python test.py --model model/model_epoch_415.pth --image butterfly_GT --scale 4 --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-SRResNet/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.


### Performance
  - We provide a pretrained model trained on [291](http://cv.snu.ac.kr/research/VDSR/train_data.zip) images with data augmentation
  - So far performance in PSNR is not as good as paper, not even comparable. Any suggestion is welcome
  
| Dataset        | SRResNet Paper          | SRResNet PyTorch|
| ------------- |:-------------:| -----:|
| Set5      | 32.05      | **30.84** |
| Set14     | 28.49      | **27.71** |
| BSD100    | 27.58      | **26.21** |

### Result
From left to right are ground truth, bicubic and SRResNet
<p>
  <img src='result/result.png' height='270' width='700'/>
</p>
