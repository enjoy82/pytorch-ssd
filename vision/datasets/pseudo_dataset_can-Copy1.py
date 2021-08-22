import numpy as np
import cv2
import glob
from torch.utils.data import Dataset
from PIL import Image
import os
import albumentations as albu
import torch


class pseudoDatasetCan(Dataset):
    def __init__(self, config, transform, backgroundPaths, canPaths, length, target_transform=None):
        self.config = config
        self.transform = transform
        self.backgroundPaths = backgroundPaths
        self.canPaths = canPaths
        self.length = length
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #疑似画像を生成する
        canPaths = self.canPaths
        backgroundPaths = self.backgroundPaths
        boxes = []
        labels = []
        img = cv2.imread(np.random.choice(backgroundPaths))
        canImg = cv2.imread(np.random.choice(canPaths))
        hr = 300/img.shape[0]
        wr = 300/img.shape[1]
        img = cv2.resize(img, (300, 300))
        canImg = cv2.resize(canImg, (int(canImg.shape[1] * wr), int(canImg.shape[0] * hr)))
        img = self._color_change(img)
        canImg = self._crop_image(canImg)
        canImg = self._crop_width(canImg)
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
        if can == True:
            r = np.random.rand() * 1.0 + 0.6
        #ターゲット画像の白以外の部分色変える処理を入れる 
        targetImg = cv2.resize(targetImg, (int(r * targetImg.shape[1]), int(r *  targetImg.shape[0])))
        if colorRotation:
            changeimg = albu.RGBShift(p = 1.0)(image=targetImg)['image']
            changeimg = albu.ChannelShuffle(p = 0.3)(image=changeimg)['image']
            targetImg = np.where(targetImg==transparence, targetImg[:, :], changeimg[:, :])
        if(backgroundImg.shape[1] <= targetImg.shape[1]):
            targetImg = targetImg[:, :backgroundImg.shape[1] - 1]
        if(backgroundImg.shape[0] <= targetImg.shape[0]):
            targetImg = targetImg[:backgroundImg.shape[0] - 1, :]
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
    
    def _crop_image(self, canImg):
        h,w,_ = canImg.shape
        if w > h:
            #上見切れるとは思えない
            r = r = np.random.rand() * 0.2
            canImg = canImg[:int(h-h*r), :]
        else:     
            if np.random.rand() < 0.45:
                r = np.random.rand() * 0.2
                canImg = canImg[:int(h-h*r), :]
            elif np.random.rand() < 0.9:
                r = np.random.rand() * 0.2
                canImg = canImg[int(h*r):h, :]
            else:
                r = np.random.rand() * 0.1
                canImg = canImg[int(h*r):int(h-h*r), :]
        return canImg
    
    def _crop_width(self, img):
        h,w,_ = img.shape
        if np.random.rand() < 0.5:
            r = np.random.rand() * 0.2
            img = img[:, :int(w-w*r)]
        else:
            r = np.random.rand() * 0.2
            img = img[:, int(w*r):]
        return img
    
    def _color_change(self, img):
        transparence = (255,255,255)
        if np.random.rand() < 0.5:
            generateImg = albu.augmentations.transforms.RGBShift(r_shift_limit = (-30, 30), g_shift_limit = (-10, 10),b_shift_limit = (-30, 30),  p = 1.0)(image=img)['image']
            img = np.where(img==transparence, img, generateImg)
        if np.random.rand() < 0.5:
            generateImg = albu.augmentations.transforms.RandomFog(p = 1.0)(image=img)['image']
            img = np.where(img==transparence, img, generateImg)
        return img