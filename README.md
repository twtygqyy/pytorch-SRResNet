# PyTorch SRResNet
Implementation of Paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"(https://arxiv.org/abs/1609.04802) in PyTorch

## Usage
### Training
```
usage: main_srresnet.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS]
                        [--lr LR] [--step STEP] [--cuda] [--resume RESUME]
                        [--start-epoch START_EPOCH] [--threads THREADS]
                        [--pretrained PRETRAINED] [--vgg_loss]

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
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --vgg_loss            Use content loss?

```
An example of training usage is shown as follows:
```
python main_srresnet.py --cuda --vgg_loss
```

### demo
```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE]
               [--dataset DATASET] [--scale SCALE]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --image IMAGE      image name
  --dataset DATASET  dataset name
  --scale SCALE      scale factor, Default: 4
```
We convert Set5 test set images to mat format using Matlab, for best PSNR performance, please use Matlab
An example of usage is shown as follows:
```
python test.py --model model/model_srresnet.pth --dataset Set5 --image butterfly_GT --scale 4 --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-SRResNet/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.


### Performance
  - We provide a pretrained model trained on [291](http://cv.snu.ac.kr/research/VDSR/train_data.zip) images with data augmentation
  - Instance Normalization is applied instead of Batch Normalization for better performance 
  - So far performance in PSNR is not as good as paper, any suggestion is welcome
  
| Dataset        | SRResNet Paper          | SRResNet PyTorch|
| ------------- |:-------------:| -----:|
| Set5      | 32.05      | **31.52** |
| Set14     | 28.49      | **27.95** |
| BSD100    | 27.58      | **27.17** |

### Result
From left to right are ground truth, bicubic and SRResNet
<p>
  <img src='result/result.png' height='270' width='700'/>
</p>
