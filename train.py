import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from models.unet import UNet
from models.unetpp import UNetpp
from models.unetppp import UNetppp
from models.attentionunet import AttUNet
from models.avnet import AVNet
from models.csnet import CSNet
import shutil
import natsort
import BatchDataReader
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torchsummary import summary
import argparse


# Global variables
DATAROOT_PATH = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\STERGE'  # Path to data
GPU_IDS = '2'  # GPU ids sa foloseasca al doilea GPU. Daca vrei mai multe le pui cu virgula  gen GPU_IDS = '0,1'
TRAIN_IDS = [0, 240]  # Train id range
VAL_IDS = [240, 250]  # Validation id range
TEST_IDS = [250, 300]  # Test id range
MODALITY_FILENAME = ['OCT(OPL_BM)', 'OCT(FULL)', 'OCT(ILM_OPL)', 'OCTA(OPL_BM)', 'OCTA(FULL)', 'OCTA(ILM_OPL)', 'GT_Multitask']  # Dataset filenames, last is label filename
DATA_SIZE = [400, 400]  # Input data size
IN_CHANNELS = 6  # Number of input channels
CHANNELS = 128  # Number of channels
SAVEROOT = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot'  # Path to save results
N_CLASSES = 5  # Number of final classes for classification
LOAD = False  # Whether to load a pre-trained model
BATCH_SIZE = 4  # Input batch size
LEARNING_RATE = 0.0005  # Initial learning rate for Adam optimizer
NUMBER_OF_EPOCHS = 200  # Number of epochs for training
MODE = 'train'  # Mode (train/test)
OPTIMIZER_TYPE = 'Adam'  # Optimizer type


def calculate_miou(predicted_img, GT_img):
    classnum = GT_img.max() #gaseste maximul care indica numarul de clase (gen artery vein etc)
    iou = np.zeros((int(classnum), 1)) #array pt a stoca iou values pt fiecare clasa
    for i in range(int(classnum)):
        imga = predicted_img == i + 1
        imgb = GT_img == i + 1
        imgi = imga * imgb
        imgu = imga + imgb
        iou[i] = np.sum(imgi) / np.sum(imgu)
    miou = np.mean(iou)
    return miou

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape = predict.shape
        target = torch.unsqueeze(target, 1)
        #implementarea la make one hot
        res = torch.zeros(shape)
        res.scatter_(1, target.long().cpu(), 1)
        target = res
        #
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = nn.functional.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


def train_net(net, device): #antreneaza neural network ul cu device ul dat (cpu sau gpu)
    # train setting
    val_num = VAL_IDS[1] - VAL_IDS[0] #numarul de valori (cate imagini de validare o sa fie)
    best_valid_miou = 0 #variabila care va contine best mean intersection over union score observed during validation
    model_save_path = os.path.join(SAVEROOT, 'checkpoints') # construieste path pentru checkpoints
    best_model_save_path = os.path.join(SAVEROOT, 'best_model') # construieste path pentru best model
    # Read Data
    train_dataset = BatchDataReader.CubeDataset(DATAROOT_PATH, TRAIN_IDS, DATA_SIZE, MODALITY_FILENAME, is_dataaug=False) #creaza training dataset folosind CubeDataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) #wrappuie datasetul de training intr un dataloader pentru a fi posibile batch processing, shuffling etc
    valid_dataset = BatchDataReader.CubeDataset(DATAROOT_PATH, VAL_IDS, DATA_SIZE, MODALITY_FILENAME,is_dataaug=False) #creaza datasetul de validare
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4) #wrappuie in data loader datasetul de validare
    # Setting Optimizer
    if OPTIMIZER_TYPE == 'SGD': #initializeaza optimizatorul (optimizatorul updateaza iterativ weights urile modelului astfel incat sa se reduca loss ul)
        optimizer = torch.optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=1e-6)
    elif OPTIMIZER_TYPE == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), LEARNING_RATE, betas=(0.9, 0.99))
    elif OPTIMIZER_TYPE == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), LEARNING_RATE, weight_decay=1e-8)
    # Setting Loss
    Loss_CE = nn.CrossEntropyLoss() #cross entropy loss function pentru classification tasks
    Loss_DSC = DiceLoss() #Dice loss function pentru segmentation tasks

    # Start train
    for epoch in range(1, NUMBER_OF_EPOCHS + 1): #itereaza asupra numarului de epochs setate
        net.train() #seteaza networkul la training mode
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader)) #wraps data loader in progres bar pentru a arata progressul
        for itr, (train_images, train_annotations, name) in pbar: #itereaza peste fiecare batch de training data
            train_images = train_images.to(device=device, dtype=torch.float32) #muta training images la cpu sau gpu (device)
            train_annotations = train_annotations.to(device=device, dtype=torch.long) #muta labels urile la cpu sau gpu
            pred = net(train_images) #passes the training images through the network to obtain predictions
            loss = Loss_CE(pred, train_annotations) + 0.6 * Loss_DSC(pred, train_annotations) #calculeaza loss ul total ca o suma ponderata dintre cross entropy loss si dice loss
            optimizer.zero_grad() #curata gradientii tuturor parametrilor optimizati pentru a preveni acumularea din pasi anteriori (iteratii anterioare cred)
            loss.backward() #calculeaza gradientul loss-ului raportat la parametrii modelului folosind backpropagation
            optimizer.step() #actualizeaza parametrii modelului bazat pe gradientii calculati
            # Start Val
        with torch.no_grad(): #da disable la calcularea gradientilor pentru a  reduce consumul de memorie si a speedui up calculele din timpul validarilor
            # Save model
            torch.save(net.module.state_dict(), os.path.join(model_save_path, f'{epoch}.pth')) #salveaza state dictionary ul modelului dupa fiecare epoch
            logging.info(f'Checkpoint {epoch} saved !')
            # Calculate validation mIOU
            val_miou_sum = 0 #initializeaza suma a miou de validare
            net.eval() # si seteaza networkul in modul de evaluare
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader)) # progress bar pt validation
            for itr, (test_images, test_annotations, cubename) in pbar: #itereaza fiece batch de validation data
                test_images = test_images.to(device=device, dtype=torch.float32) #muta imaginile de test la device si converteste adnotarile la numpy array
                test_annotations = test_annotations.cpu().detach().numpy() # muta labels la device si converteste adnotarile la numpy array
                pred = net(test_images) #TRECE VALIDATION IMAGES PRIN NETWORK PENTRU A OBTINE PREDICTII
                pred_argmax = torch.argmax(pred, dim=1) #calculeaza predicted class for each pixel
                result = np.squeeze(pred_argmax).cpu().detach().numpy() #converteste predicted class la numpy array
                val_miou_sum += calculate_miou(result, test_annotations) #acumuleaza miou pentru validation batch
            val_miou = val_miou_sum / val_num #calculeaza average miou pentru validation set
            print("Step:{}, Valid_mIoU:{}".format(epoch, val_miou))

            # save best model (daca validation miou al epoch ului curent este cel mai bun de pana acum, se salveaza modelul ca best model, se creaza un director cu numele
            if val_miou > best_valid_miou:# fiind valoarea miou, se copiaza checkpoint ul la acest director. Se pastreaza doar ultimele 3 cele mai bune modele, stergand restul
                temp = '{:.6f}'.format(val_miou)
                os.mkdir(os.path.join(best_model_save_path, temp))
                temp2 = f'{epoch}.pth'
                shutil.copy(os.path.join(model_save_path, temp2), os.path.join(best_model_save_path, temp, temp2))
                model_names = natsort.natsorted(os.listdir(best_model_save_path))
                if len(model_names) == 4:
                    shutil.rmtree(os.path.join(best_model_save_path, model_names[0]))
                best_valid_miou = val_miou


if __name__ == '__main__':
    # setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # loading options
    # setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS #controleaza ce GPUs sunt vizibile pentru script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #vede daca cpu sau gpu
    logging.info(f'Using device {device}')
    # loading network
    net = UNet(in_channels=IN_CHANNELS, n_classes=N_CLASSES, channels=CHANNELS) #initializeaza o instanta a unet-ului cu param specificati
    # net = AVNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetpp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetppp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = AttUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = CSNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    # net=torch.nn.DataParallel(net,[0]).cuda()
    net = torch.nn.DataParallel(net, [0]) #wrappuie modelul cu dataparellel pentru a enable-ui multi GPU training. Se specifica sa se foloseasca doar GPU cu id 0???????????
    # load trained model
    if LOAD: #incarca state dictionary ul unui model pre antrenat daca LOAD e setat la un path valid  dar e pus False acum
        net.load_state_dict(
            torch.load(LOAD, map_location=device)
        )
        logging.info(f'Model loaded from {LOAD}')
    # input the model into GPU
    # net.to(device=device)
    try:
        train_net(net=net, device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
