import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import _NetG, _NetD
from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRGAN")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default=0.0001")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

def main():

    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("/path/to/your/hdf5/data/like/rgb_srresnet_x4.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)

    print("===> Loading VGG model")
    netVGG = models.vgg19()
    netVGG.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"))
    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

        def forward(self, x):
            out = self.feature(x)
            return out

    print("===> Building content model")
    netContent = _content_model()

    print("===> Building generator model")
    netG = _NetG()

    print("===> Building discriminator model")    
    netD = _NetD()

    print("===> Building criterions")  
    mse_criterion = nn.MSELoss(size_average=False)
    entropy_criterion = nn.BCELoss()

    print("===> Setting GPU")
    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        netContent = netContent.cuda()
        mse_criterion = mse_criterion.cuda()
        entropy_criterion = entropy_criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizerG, optimizerD, netG, \
            netD, netContent, mse_criterion, entropy_criterion, epoch)
        save_checkpoint(netG, epoch)

def total_gradient(parameters):
    """Computes a gradient clipping coefficient based on gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters: 
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1./2)
    return totalnorm
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr    

def train(training_data_loader, optimizerG, optimizerD, netG, netD, netContent, mse_criterion, entropy_criterion, epoch):

    lrG = adjust_learning_rate(optimizerG, epoch-1)
    lrD = adjust_learning_rate(optimizerD, epoch-1)

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lrG

    for param_group in optimizerD.param_groups:
        param_group["lr"] = lrD

    print "epoch =", epoch," lrG =",optimizerG.param_groups[0]["lr"], \
        "lrD =",optimizerD.param_groups[0]["lr"]

    netG.train()
    netD.train()
    one = torch.FloatTensor([1.])
    mone = one * -1
    content_weight = torch.FloatTensor([1.])
    adversarial_weight = torch.FloatTensor([0.001])

    real_label = 1
    fake_label = 0    

    for iteration, batch in enumerate(training_data_loader, 1):

        input, real = Variable(batch[0]), Variable(batch[1])
        label = Variable(torch.FloatTensor(opt.batchSize))

        if opt.cuda:
            input = input.cuda()
            real = real.cuda()
            label = label.cuda()
            one, mone, content_weight, adversarial_weight = \
                one.cuda(), mone.cuda(), content_weight.cuda(), adversarial_weight.cuda()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        for p in netD.parameters(): # reset requires_grad
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        # train with real
        label.data.resize_(opt.batchSize).fill_(real_label)
        output = netD(real)
        errD_real = entropy_criterion(output, label)
        errD_real.backward()

        # train with fake
        #input_G = Variable(input.data, volatile = True)
        fake = netG(input)
        #fake_D = Variable(netG(input_G).data)
        label.data.fill_(fake_label)

        output = netD(fake.detach())
        errD_fake = entropy_criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        netContent.zero_grad() 

        fake = netG(input)
        content_fake = netContent(fake)
        content_real = netContent(real)
        content_real = Variable(content_real.data)

        content_loss = mse_criterion(content_fake, content_real)
        content_loss.backward(content_weight, retain_graph=True)

        label.data.fill_(real_label)
        output = netD(fake)
        errG = entropy_criterion(output, label)
        errG.backward(adversarial_weight)

        optimizerG.step()

        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): LossD: {:.10f} [{:.10f} - {:.10f}] LossG: [{:.5f} + {:.5f}]".format(epoch, iteration, len(training_data_loader), 
                  errD.data[0], errD_real.data[0], errD_fake.data[0], errG.data[0], content_loss.data[0]))            
            print "gradient_D:", total_gradient(netD.parameters()), "gradient_G:", total_gradient(netG.parameters())

def save_checkpoint(model, epoch):
    model_out_path = "srgan_checkpoint/" + "srgan_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("srgan_checkpoint/"):
        os.makedirs("srgan_checkpoint/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
