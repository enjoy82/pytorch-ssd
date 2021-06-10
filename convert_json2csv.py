import json
import cv2
import os
import csv
import numpy as np

#openimageの項目,そのまま使いまわせるように
openImageRow = ["ImageID","LabelName","Confidence","XMin","XMax","YMin","YMax","ClassName"]

#jsonファイル指定
jsonFiles = [""]

#画像ある場所
allImagesPath = "./all_images"

#準備
test_csv = open(os.path.join(allImagesPath, "sub-test-annotations-bbox.csv"), "w")
test_writer = csv.writer(test_csv, lineterminator='\n')
test_writer.writerow(openImageRow)
train_csv = open(os.path.join(allImagesPath,"sub-train-annotations-bbox.csv"), "w")
train_writer = csv.writer(train_csv, lineterminator='\n')
train_writer.writerow(openImageRow)
val_csv = open(os.path.join(allImagesPath,"sub-validation-annotations-bbox.csv"), "w")
val_writer = csv.writer(val_csv, lineterminator='\n')
val_writer.writerow(openImageRow)

writers = [train_writer, test_writer, val_writer]
type_names = ["train", "test", "validation"]

for path in jsonFiles:
    jsonFile = json.load(open(path, 'r'))

    typeidx = 0
    rand = np.random.rand()
    if rand < 0.1:
        typeidx = 1
    elif rand > 0.9:
        typeidx = 2

    for information in jsonFile.values():
        filename = os.path.join(allImagesPath,information["filename"])
        img_org = cv2.imread(filename)
        h, w = img_org.shape[:2]

        regions = information["regions"]
        for region in regions:
            objectName = region["region_attributes"]["name"].replace( '\n' , '') #改行消す
            shape = region["shape_attributes"]
            x1 = shape["x"]
            x2 = x1 + shape["width"]
            y1 = shape["y"]
            y2 = y1 + shape["height"]

            row = [filename,"takadamaLab_carryber",1.0,x1/h,x2/h,y1/w,y2/w,objectName]
            writers[typeidx].writerow(row)
