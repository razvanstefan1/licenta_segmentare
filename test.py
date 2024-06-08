import torch
import torch.nn as nn
import logging
import sys
import os
from models.unet import UNet
from models.unetpp import UNetpp
from models.unetppp import UNetppp
from models.attentionunet import AttUNet
from models.avnet import AVNet
from models.csnet import CSNet
import numpy as np

from options.test_options import TestOptions
import cv2
import natsort
import BatchDataReader
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

def test_net(net,device):

    test_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_results = os.path.join(opt.saveroot, 'test_results')
    BGR=np.zeros((opt.data_size[0],opt.data_size[1],3))
    net.eval()

    #test set
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
    for itr, (test_images, test_annotations,cubename) in pbar:
        test_images = test_images.to(device=device, dtype=torch.float32)
        pred= net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.cpu().detach().numpy()
        print(cubename[0])
        cv2.imwrite(os.path.join(test_results, cubename[0]),pred_argmax[0,:,:])
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_softmax = pred_softmax.cpu().detach().numpy()
        BGR[:,:,0]=53*pred_softmax[0,0,:,:]+143*pred_softmax[0,1,:,:]+28*pred_softmax[0,2,:,:]+186*pred_softmax[0,3,:,:]+106*pred_softmax[0,4,:,:]
        BGR[:,:,1]=32*pred_softmax[0,0,:,:]+165*pred_softmax[0,1,:,:]+25*pred_softmax[0,2,:,:]+131*pred_softmax[0,3,:,:]+217*pred_softmax[0,4,:,:]
        BGR[:,:,2]=15*pred_softmax[0,0,:,:]+171*pred_softmax[0,1,:,:]+215*pred_softmax[0,2,:,:]+43*pred_softmax[0,3,:,:]+166*pred_softmax[0,4,:,:]
        cv2.imwrite(os.path.join(opt.saveroot, 'test_visuals', cubename[0]), BGR)


if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    net= UNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #net = AVNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #net = UNetpp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #net = UNetppp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #net = AttUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #net = CSNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    #load trained model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(opt.saveroot, 'best_model',
                                natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1]) + '/' + \
                   os.listdir(bestmodelpath)[0]
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
