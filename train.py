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


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\STERGE',
                        help='path to data')
    parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids')
    parser.add_argument('--train_ids', type=list, default=[0, 240], help='train id number')
    parser.add_argument('--val_ids', type=list, default=[240, 250], help='val id number')
    parser.add_argument('--test_ids', type=list, default=[250, 300], help='test id number')
    parser.add_argument('--modality_filename', type=list,
                        default=['OCT(OPL_BM)', 'OCT(FULL)', 'OCT(ILM_OPL)', 'OCTA(OPL_BM)', 'OCTA(FULL)',
                                 'OCTA(ILM_OPL)', 'GT_Multitask'], help='dataset filename, last name is label filename')
    parser.add_argument('--data_size', type=list, default=[400, 400], help='input data size separated with comma')
    parser.add_argument('--in_channels', type=int, default=6, help='input channels')
    parser.add_argument('--channels', type=int, default=128, help='channels')
    parser.add_argument('--saveroot',
                        default='C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot',
                        help='path to save results')
    parser.add_argument('--n_classes', type=int, default=5, help='final class number for classification')
    parser.add_argument('--load', type=str, default=False, help='whether restore or not')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
    parser.add_argument('--num_epochs', type=int, default=200, help='iterations for batch_size samples')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--optimizer', type=str, default='Adam')

    opt = parser.parse_args()
    return opt


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


def train_net(net, device, opt):
    # train setting
    val_num = opt.val_ids[1] - opt.val_ids[0]
    best_valid_miou = 0
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    # Read Data
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids, opt.data_size, opt.modality_filename,
                                                is_dataaug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,
                                                is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    # Setting Loss
    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC = DiceLoss()
    # Start train
    for epoch in range(1, opt.num_epochs + 1):
        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations, name) in pbar:
            train_images = train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            pred = net(train_images)
            loss = Loss_CE(pred, train_annotations) + 0.6 * Loss_DSC(pred, train_annotations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Start Val
        with torch.no_grad():
            # Save model
            torch.save(net.module.state_dict(), os.path.join(model_save_path, f'{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved !')
            # Calculate validation mIOU
            val_miou_sum = 0
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations, cubename) in pbar:
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = test_annotations.cpu().detach().numpy()
                pred = net(test_images)
                pred_argmax = torch.argmax(pred, dim=1)
                result = np.squeeze(pred_argmax).cpu().detach().numpy()
                val_miou_sum += calculate_miou(result, test_annotations)
            val_miou = val_miou_sum / val_num
            print("Step:{}, Valid_mIoU:{}".format(epoch, val_miou))
            # save best model
            if val_miou > best_valid_miou:
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
    opt = parse_options()
    # setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # loading network
    net = UNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = AVNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetpp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetppp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = AttUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = CSNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    # net=torch.nn.DataParallel(net,[0]).cuda()
    net = torch.nn.DataParallel(net, [0])
    # load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    # input the model into GPU
    # net.to(device=device)
    try:
        train_net(net=net, device=device, opt=opt)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
