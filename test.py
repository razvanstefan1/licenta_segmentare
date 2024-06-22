import argparse
import logging
import os
import sys

import cv2
import natsort
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

import BatchDataReader
from models.unet import UNet


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
    parser.add_argument('--mode', type=str, default='test')

    opt = parser.parse_args()
    return opt


def test_net(net, device, opt):
    test_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,
                                               is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_results = os.path.join(opt.saveroot, 'test_results')
    BGR = np.zeros((opt.data_size[0], opt.data_size[1], 3))
    net.eval()

    # Test set
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
    for itr, (test_images, test_annotations, cubename) in pbar:
        test_images = test_images.to(device=device, dtype=torch.float32)
        pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.cpu().detach().numpy()
        print(cubename[0])
        cv2.imwrite(os.path.join(test_results, cubename[0]), pred_argmax[0, :, :])
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_softmax = pred_softmax.cpu().detach().numpy()
        BGR[:, :, 0] = 53 * pred_softmax[0, 0, :, :] + 143 * pred_softmax[0, 1, :, :] + 28 * pred_softmax[0, 2, :,
                                                                                             :] + 186 * pred_softmax[0,
                                                                                                        3, :,
                                                                                                        :] + 106 * pred_softmax[
                                                                                                                   0, 4,
                                                                                                                   :, :]
        BGR[:, :, 1] = 32 * pred_softmax[0, 0, :, :] + 165 * pred_softmax[0, 1, :, :] + 25 * pred_softmax[0, 2, :,
                                                                                             :] + 131 * pred_softmax[0,
                                                                                                        3, :,
                                                                                                        :] + 217 * pred_softmax[
                                                                                                                   0, 4,
                                                                                                                   :, :]
        BGR[:, :, 2] = 15 * pred_softmax[0, 0, :, :] + 171 * pred_softmax[0, 1, :, :] + 215 * pred_softmax[0, 2, :,
                                                                                              :] + 43 * pred_softmax[0,
                                                                                                        3, :,
                                                                                                        :] + 166 * pred_softmax[
                                                                                                                   0, 4,
                                                                                                                   :, :]
        cv2.imwrite(os.path.join(opt.saveroot, 'test_visuals', cubename[0]), BGR)


if __name__ == '__main__':
    # Setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Loading options
    opt = parse_options()
    # Setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # Loading network
    net = UNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = AVNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetpp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = UNetppp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = AttUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    # net = CSNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    # Load trained model
    best_model_path = os.path.join(opt.saveroot, 'best_model',
                                   natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(best_model_path, os.listdir(best_model_path)[0])
    net.load_state_dict(torch.load(restore_path, map_location=device))
    # Input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net, device=device, opt=opt)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
