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

DATAROOT = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\MODALITIES'# Path to data
GPU_IDS = '2'# GPU IDs
TRAIN_IDS = [0, 240]# Train ID numbers
VAL_IDS = [240, 250]# Validation ID numbers
TEST_IDS = [250, 300]# Test ID numbers
MODALITY_FILENAME = ['OCT(OPL_BM)', 'OCT(FULL)', 'OCT(ILM_OPL)', 'OCTA(OPL_BM)', 'OCTA(FULL)', 'OCTA(ILM_OPL)', 'GT_Multitask']# Dataset filenames, last name is label filename
DATA_SIZE = [400, 400]# Input data size separated by comma
IN_CHANNELS = 6# Input channels
CHANNELS = 128# Channels  (each convolutional layer o sa aiba 128 de filtere gen kernel uri de alea carora le dai slide peste fiecare pixel)
SAVEROOT = 'C:\\Users\\brolz\\Desktop\\FACULTATE\\LICENTA\\COD_LICENTA_SEGMENTARE\\saveroot'# Path to save results
N_CLASSES = 5# Final class number for classification gen cate clase sa prezica (de la 0 la 4)
MODE = 'test'# Mode (e.g., 'test', 'train')

def test_net(net, device): #functia pentru testat netul
    test_dataset = BatchDataReader.CubeDataset(DATAROOT, TEST_IDS, DATA_SIZE, MODALITY_FILENAME, is_dataaug=False) #obtinem un dataset pentru test folosind cubedataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1) #se creaza un data loader cu batch size 1
    test_results = os.path.join(SAVEROOT, 'test_results') #path ul unde se salveaza rezultatele testelor
    BGR = np.zeros((DATA_SIZE[0], DATA_SIZE[1], 3)) #numpy array gol pentru  a stoca BGR data
    net.eval()   #seteaza neural network ul sa fie pe evaluation mode

    # Test set
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader)) #progress bar, total este numarul de iteratii pentru progress bar
    for itr, (test_images, test_annotations, cubename) in pbar: #iterates over each batch of test data, cubename e filenames for the current batch
        test_images = test_images.to(device=device, dtype=torch.float32) #se muta imaginile de test pe device_ul specificat (cpu sau gpu)
        pred = net(test_images) # obtine predictii de la network net(test_images) trece imaginile prin neural network pt a obtine predictiile
        pred_argmax = torch.argmax(pred, dim=1) #se computeaza indexul valorii maxime pe dimensiunea 1, care corespunde clasei prezise (C A V F)
        pred_argmax = pred_argmax.cpu().detach().numpy() #muta tensor ul (matricea) la CPU si o detaseaza de computation graph si o transf in numpy array
        print(cubename[0]) #se printeaza filename al imaginii care e testata
        cv2.imwrite(os.path.join(test_results, cubename[0]), pred_argmax[0, :, :]) #se salveaza predicted class map ca imagine
        pred_softmax = torch.nn.functional.softmax(pred, dim=1) #aplica softmax pe predictii pentru a obtine class probabilities
        pred_softmax = pred_softmax.cpu().detach().numpy() # muta softmax probabilities la cpu, le detaseaza de computation graph si le transforma in numpy array

        #LINIILE DE MAI JOS OFERA SMOOTH TRANSITIONS LA CULORI IN FCT DE PROBABILITATI, de aia erau 10000 de culori diferite cand faceam histograma
                    #softmax probabilities of class 0 are multiplied by 53, for class 1 by 143 and so on
        BGR[:,:,0]=53*pred_softmax[0,0,:,:]+143*pred_softmax[0,1,:,:]+28*pred_softmax[0,2,:,:]+186*pred_softmax[0,3,:,:]+106*pred_softmax[0,4,:,:] #blue channel e creat combinand probabilitatile fiecarei clase, weighted cu coeficientii 53, 143, 28, 186, 106
        BGR[:,:,1]=32*pred_softmax[0,0,:,:]+165*pred_softmax[0,1,:,:]+25*pred_softmax[0,2,:,:]+131*pred_softmax[0,3,:,:]+217*pred_softmax[0,4,:,:] # green channel e creat similar, cu coef diferiti 32, 165, 25, 131, 217
        BGR[:,:,2]=15*pred_softmax[0,0,:,:]+171*pred_softmax[0,1,:,:]+215*pred_softmax[0,2,:,:]+43*pred_softmax[0,3,:,:]+166*pred_softmax[0,4,:,:] # red channel e creat similar cu coeficientii 15, 171, 215, 43, 166

        cv2.imwrite(os.path.join(SAVEROOT, 'test_visuals', cubename[0]), BGR) #genereaza imaginea cu culori


if __name__ == '__main__':
    # Setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS #restrictioneaza gpu urile vizibile la cele specificate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #daca e un gpu available il pune pe ala, daca nu, pe CPU
    logging.info(f'Using device {device}')

    # Loading network
    net = UNet(in_channels=IN_CHANNELS, n_classes=N_CLASSES, channels=CHANNELS) #initializeaza o instanta a modelului cu input channels specificate, nr de clase specificate si nr de canale

    # Load trained model
    best_model_path = os.path.join(SAVEROOT, 'best_model',natsort.natsorted(os.listdir(os.path.join(SAVEROOT, 'best_model')))[-1]) #face path pt best models si alege ultimul model directory dupa sortare (cel mai bun)
    restore_path = os.path.join(best_model_path, os.listdir(best_model_path)[0]) #selecteaza path ul la primul fisier din best model directory
    net.load_state_dict(torch.load(restore_path, map_location=device)) #loads model state from restore_path (path ul spre modelul efectiv care va fi folosit)
    # Input the model into GPU
    net.to(device=device) #muta modelul la device ul specificat pentru ca operatiile urmatoare sa fie efectuate pe acel device
    try:
        test_net(net=net, device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
