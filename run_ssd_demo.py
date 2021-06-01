import cv2
import numpy as np
import time
from utils_for_openvino.mb3lite_predictor import create_mobilenetv3_small_ssd_lite_predictor
 
# モジュール読み込み 
import sys
sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin

windowwidth = 320
windowheight = 240
image_sige = 300
nms_method = "hard"
label_path = "./models/open-images-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]


# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み 
net = IENetwork(model='./models/mbv3-ssd-v1.xml', weights='./models/mbv3-ssd-v1.bin')
exec_net = plugin.load(network=net)
input_blob_name = list(net.inputs.keys())[0]
output_blob_name = sorted(list(net.outputs.keys()))

#predictor
predictor = create_mobilenetv3_small_ssd_lite_predictor(exec_net, image_size = image_sige,  nms_method=nms_method, input = input_blob_name, output = output_blob_name)

#print("stand", input_blob_name, output_blob_name)
# カメラ準備 
#cap = cv2.VideoCapture(0)
"""
if cap.isOpened() != True:
    print("camera open error!")
    quit()
else:
    print("camera open!")
"""

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, windowwidth)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, windowheight)

# メインループ 
while True:
    #ret, frame = cap.read()
    # Reload on error 
    #if ret == False:
    #    continue
    frame = cv2.imread("./gun.jpg")
    boxes, labels, probs = predictor.predict(frame,10, 0.4) #TODO 閾値
    # 出力から必要なデータのみ取り出し 
    #TODO label怪しい
    print(boxes, labels, probs)

    boxed = [] #重複box
    for i in range(len(boxes)):
        box = boxes[i, :]
        box = list(map(int, box))
        flag = 1
        for b in boxed:#重複チェック
            if np.all(box == b):
                flag = 0
        if flag == 0:
            continue
        boxed.append(box)

        label = class_names[labels[i]] + str(probs[i])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.putText(frame, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    
    # 画像表示 
    cv2.imshow('frame', frame)
    # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    break
    if key != -1:
        break
    
# 終了処理 
#cap.release()
cv2.destroyAllWindows()
