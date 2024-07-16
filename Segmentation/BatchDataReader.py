import os
import natsort
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2
import random
from skimage import transform
from skimage.util import random_noise
from skimage.filters import gaussian


class CubeDataset(Dataset):
    # data_dir = root directory ce contine datasetul
    # data_id = tupla ce specifica range ul indicilor datelor de folosit
    # data_size = dimensiunea imaginilor (tupla)
    # modality = lista cu numele directoarelor ce le ia ca input gen opl_ibm etc
    # is_Dataaug daca sa se aplice augmentarea datelor

    # pentru fiecare modality in afara de ultima (care e label), se colecteaz caile spre iamgini si se stocheaza in datasetlist[data] (dictionar)
    def __init__(self, data_dir, data_id, data_size, modality, is_dataaug=True):
        self.is_dataaug = is_dataaug
        self.datanum = data_id[1] - data_id[0]  # numarul de data samples
        self.modality = modality
        self.data_size = data_size
        self.modalitynum = len(modality) - 1  # numarul total de modalities (cu tot cu label)
        self.datasetlist = {'data': {}, 'label': {}}
        for modal in modality:  # itereaza peste fiecare modality si umple datasetlist cu caile spre imagini
            if modal != modality[-1]:  # ne asiguram ca nu e label modality (care e ultima)
                self.datasetlist['data'].update(
                    {modal: {}})  # adauga un nou dictionar pt modalitatea curenta pt a stoca image paths
                imglist = os.listdir(os.path.join(data_dir, modal))  # listeaza toate fisierele din directorul modalitatii curente si le salveaza in imglist
                imglist = natsort.natsorted(imglist)  # le sorteaza in ordine naturala (de la 10001 la 10300 sau cate pui)
                for img in imglist[data_id[0]:data_id[
                    1]]:  # itereaza peste imaginile sortate din range ul specificat de tupla data_id
                    self.datasetlist['data'][modal].update(
                        {img: {}})  # initializeaza un dictionar pentru fiecare imagine
                    imgadress = os.path.join(data_dir, modal, img)  # contruieste full path la imagine
                    self.datasetlist['data'][modal][img] = imgadress  # pune image path in dictionary
            else:  # block pentru label modality
                imglist = os.listdir(os.path.join(data_dir, modal))  # listeaza toate fisierele din label directory
                imglist = natsort.natsorted(imglist)  # sorteaza in ordine naturala
                for img in imglist[data_id[0]:data_id[1]]:  # itereaza peste lista sortata de labels
                    self.datasetlist['label'].update(
                        {img: {}})  # initializeaza o intrare in dictionar pentru fiecare label image
                    labeladdress = os.path.join(data_dir, modal, img)  # construieste full path la label image
                    self.datasetlist['label'][img] = labeladdress  # stocheaza label image apth in dictionar

    def __getitem__(self,
                    index):  # metoda care face fetch la o singura imagine si label ul ei (dupa index ul dat ca param)
        data = np.zeros((self.modalitynum, self.data_size[0], self.data_size[
            1]))  # array numpy initializat cu 0 penrtu a tine toate modalities ale unei imagini. e ca si cum ai avea 6 matrice de 400 pe 400 (6 imagini) stocate in array, dar pt o singura imagine
        label = np.zeros((self.data_size[0], self.data_size[1]))  # matrice ce tine label ul initializata cu 0
        for i, modal in enumerate(self.modality):  # itereaza peste fiecare modalitate a imaginii
            if modal != self.modality[-1]:  # verifica sa nu fie label modality
                name = list(self.datasetlist['data'][modal])[
                    index]  # primeste numele imaginii de la index ul dat si modalitatea curenta
                data[i, :, :] = cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE).astype(
                    np.float32)  # incarca imaginea ca fiind grayscale si o converteste la array numpy de tip float32. Imaginea e stocata in the 'data' array
            else:
                name = list(self.datasetlist['label'])[index]  # retrieves numele label image ului la indexul dat
                label[:, :] = cv2.imread(self.datasetlist['label'][name], cv2.IMREAD_GRAYSCALE).astype(
                    np.float32)  # incarca labelul ca pe un grayscale si converteste in numpy array. Imaginea e stocata in the 'label' array

        # data augmentation
        if self.is_dataaug == True:  # daca e enabled data augmentation, se aplica metoda augmentation pe data si label al imaginii data ca parametru prin index
            data, label = self.augmentation(data, label)  # apel metoda augmentare
        data = torch.from_numpy(np.ascontiguousarray(
            data))  # se converteste data si label la tensor pytorch. contiguous array se asigura ca array ul e stocat in memorie contigua pentru computare eficienta
        label = torch.from_numpy(np.ascontiguousarray(label))
        return data, label, name  # se returneaza tensor ul cu imaginea si label ul si numele imaginii

    # image augmentation
    def augmentation(self, image, annotation):
        # rotate
        if torch.randint(0, 4, (1,)) == 0: #sansa de 25% sa se roteasca imaginea si label ul cu un unghi random intre -30 si 30 de grade (toate modalities se rotesc, evident)
            angle = torch.randint(-30, 30, (1,))
            for i in range(self.modalitynum):
                image[i, :, :] = transform.rotate(image[i, :, :], angle)
            annotation = transform.rotate(annotation, angle)
        # flipud #sansa sa se faca flip vertical
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum): #iteram peste toate modalities in afara de label
                image[i, :, :] = np.flipud(image[i, :, :]) # 'image[i, :, :]' reprezinta slice-ul 2D (modalitatea) a i-a. operatorul : selecteaza toate liniile si coloanele ( toti pixelii imaginii)
            annotation = np.flipud(annotation) #flip la label (annotation = label)
        # fliplr #sansa sa se faca orizontal
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i, :, :] = np.fliplr(image[i, :, :])
            annotation = np.fliplr(annotation)
            '''
        #noise #sansa sa se adauge gaussian noise
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = random_noise(image[i,:,:], mode='gaussian')
        '''
        return image, annotation

    def __len__(self):
        return self.datanum