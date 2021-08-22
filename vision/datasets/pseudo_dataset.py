import numpy as np
import cv2
import glob
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as albu
import torch

import numpy as np
import cv2
import glob
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as albu

class pseudoDataset(Dataset):
    def __init__(self, config, transform, backgroundPaths, conePaths, canPaths, length, target_transform=None):
        self.config = config
        self.transform = transform
        self.backgroundPaths = backgroundPaths
        self.conePaths = conePaths
        self.canPaths = canPaths
        self.length = length
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #疑似画像を生成する
        canPaths = self.canPaths
        conePaths = self.conePaths
        backgroundPaths = self.backgroundPaths
        boxes = []
        labels = []
        img = cv2.imread(np.random.choice(backgroundPaths))
        hr = 300/img.shape[0]
        wr = 300/img.shape[1]
        img = cv2.resize(img, (300, 300))
        img = self._color_change(img)
        """
        if np.random.rand() < 1.0:
            #コーン&缶
            print("double")
            coneImg = cv2.imread(np.random.choice(conePaths))
            canImg = cv2.imread(np.random.choice(canPaths))
            miximg, rxmin, rxmax, rymin, rymax = self._miximg(coneImg, canImg)
            img, xmin, xmax, ymin, ymax, r = self._randomPasteImg(img, miximg, can = True)
            labels.append(2)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
            xxmin = (rxmin * r + xmin * img.shape[1]) / img.shape[1]
            yymin = (rymin * r + ymin * img.shape[0]) / img.shape[0]
            xxmax = (rxmax * r + xmin * img.shape[1]) / img.shape[1]
            yymax = (rymax * r + ymin * img.shape[0]) / img.shape[0]
            boxes.append([xxmin, yymin, xxmax, yymax])
        """
        if np.random.rand() <= 2.0:
            coneImg = cv2.imread(np.random.choice(conePaths))
            coneImg = cv2.resize(coneImg, (int(coneImg.shape[1] * wr), int(coneImg.shape[0] * hr)))
            img, xmin, xmax, ymin, ymax, _ = self._randomPasteImg(img, coneImg, cone = True)
            labels.append(2)
            boxes.append([xmin, ymin, xmax, ymax])
        else:
            canImg = cv2.imread(np.random.choice(canPaths))
            canImg = cv2.resize(canImg, (int(canImg.shape[1] * wr), int(canImg.shape[0] * hr)))
            canImg = self._color_change(canImg)
            img, xmin, xmax, ymin, ymax, _  = self._randomPasteImg(img, canImg, colorRotation = True, can = True)
            labels.append(1)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = np.array(boxes)
        labels = np.array(labels)
        #self._displayImg(img, boxes, labels)
        img, boxes, labels = self._getitem(img, boxes, labels)
        return img, boxes, labels

    def _getitem(self, img, boxes, labels):
        img = cv2.resize(img, (300, 300))
        boxes[:, 0] *= img.shape[1]
        boxes[:, 1] *= img.shape[0]
        boxes[:, 2] *= img.shape[1]
        boxes[:, 3] *= img.shape[0]
        img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        else:
            boxes = torch.from_numpy(boxes)
            labels = torch.from_numpy(labels)
        boxes = boxes.to(torch.float32)
        labels = labels.to(torch.int64)
        return img, boxes, labels

    def _randomPasteImg(self, backgroundImg, targetImg, colorRotation = False, cone = False, can = False):
        #いい感じにペースト処理をする
        transparence = (255,255,255)
        if cone == True:
            r = np.random.rand() * 1.0 + 0.3
        elif can == True:
            r = np.random.rand() * 1.0 + 0.6
        #ターゲット画像の白以外の部分色変える処理を入れる 
        targetImg = cv2.resize(targetImg, (int(r * targetImg.shape[1]), int(r *  targetImg.shape[0])))
        #画像合わせる
        if backgroundImg.shape[1] - targetImg.shape[1] <= 1:
            targetImg = targetImg[:, :backgroundImg.shape[1]-1]
        if backgroundImg.shape[0] - targetImg.shape[0] <= 1:
            targetImg = targetImg[:backgroundImg.shape[0]-1, :]
            
        if colorRotation:
            changeimg = albu.RGBShift(p = 1.0)(image=targetImg)['image']
            changeimg = albu.ChannelShuffle(p = 0.3)(image=changeimg)['image']
            targetImg = np.where(targetImg==transparence, targetImg[:, :], changeimg[:, :])
        sx = np.random.randint(0, backgroundImg.shape[1] - targetImg.shape[1])
        sy = np.random.randint(0, backgroundImg.shape[0] - targetImg.shape[0])
        miximg = np.where(targetImg==transparence, backgroundImg[sy:sy+targetImg.shape[0], sx:sx+targetImg.shape[1]], targetImg)
        backgroundImg[sy:sy+targetImg.shape[0], sx:sx+targetImg.shape[1]] = miximg
        xmin = sx / backgroundImg.shape[1]
        xmax = (sx+targetImg.shape[1]) / backgroundImg.shape[1]
        ymin = sy / backgroundImg.shape[0]
        ymax = (sy+targetImg.shape[0]) / backgroundImg.shape[0]
        
        if np.random.rand() < 0.2:
            backgroundImg = cv2.GaussianBlur(backgroundImg,(3,3),0)
        return backgroundImg, xmin, xmax, ymin, ymax, r
    
    def _displayImg(self, img, boxes, labels):
        img = cv2.resize(img, (300, 300))
        boxes[:, 0] *= img.shape[1]
        boxes[:, 1] *= img.shape[0]
        boxes[:, 2] *= img.shape[1]
        boxes[:, 3] *= img.shape[0]
        for i, box in enumerate(boxes):
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0),  thickness=8)
            cv2.putText(img, str(labels[i]),
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        display(Image.fromarray(img))
    
    def _miximg(self, coneimg, canimg):
        transparence = (255,255,255)
        changeimg = albu.RGBShift(p = 1.0)(image=canimg)['image']
        changeimg = albu.ChannelShuffle(p = 0.3)(image=changeimg)['image']
        canimg = np.where(canimg==transparence, canimg[:, :], changeimg[:, :])
        xrate = coneimg.shape[1] / canimg.shape[1]
        yrate = coneimg.shape[0] / canimg.shape[0]
        #二倍以上はほしい
        if xrate < 2.0 or yrate < 2.0:
            r = max(0, np.random.rand() * coneimg.shape[1] / (2.0 * canimg.shape[1]) - 0.4) + 0.4
            canimg = cv2.resize(canimg, (int(r * canimg.shape[1]), int(r * canimg.shape[0])))
        print(canimg.shape, coneimg.shape)
        sx = np.random.randint(0, coneimg.shape[1] - canimg.shape[1])
        sy = np.random.randint(max(0, coneimg.shape[0] - canimg.shape[0] - 50), coneimg.shape[0] - canimg.shape[0])
        #miximg = np.where(canimg==transparence, coneimg[sy:sy+canimg.shape[0], sx:sx+canimg.shape[1]], canimg)
        coneimg[sy:sy+canimg.shape[0], sx:sx+canimg.shape[1]] = canimg
        #display(Image.fromarray(coneimg))
        return coneimg, sx, sx+canimg.shape[1], sy, sy+canimg.shape[0]
    
    def _color_change(self, img):
        transparence = (255,255,255)
        if np.random.rand() < 0.5:
            generateImg = albu.augmentations.transforms.RGBShift(r_shift_limit = (-30, 30), g_shift_limit = (-10, 10),b_shift_limit = (-30, 30),  p = 1.0)(image=img)['image']
            img = np.where(img==transparence, img, generateImg)
        if np.random.rand() < 0.5:
            generateImg = albu.augmentations.transforms.RandomFog(p = 1.0)(image=img)['image']
            img = np.where(img==transparence, img, generateImg)
        return img