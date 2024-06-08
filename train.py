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
import utils
import shutil
import natsort
from options.train_options import TrainOptions
import BatchDataReader
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torchsummary import summary



def train_net(net,device):
    #train setting
    val_num = opt.val_ids[1] - opt.val_ids[0]
    best_valid_miou=0
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    # Read Data
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,is_dataaug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    #Setting Loss
    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC= utils.DiceLoss()
    #Start train
    for epoch in range(1, opt.num_epochs + 1):
        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations, name) in pbar:
            train_images =train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            pred= net(train_images)
            loss = Loss_CE(pred, train_annotations)+0.6*Loss_DSC(pred, train_annotations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Start Val
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
                result= np.squeeze(pred_argmax).cpu().detach().numpy()
                val_miou_sum += utils.cal_miou(result, test_annotations)
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
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TrainOptions().parse()
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
    #net=torch.nn.DataParallel(net,[0]).cuda()
    net=torch.nn.DataParallel(net,[0])
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    #input the model into GPU
    #net.to(device=device)
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




