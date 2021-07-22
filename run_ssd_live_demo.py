import cv2
import numpy as np
import time
from utils_for_openvino.mb3lite_predictor import create_mobilenetv3_small_ssd_lite_predictor
 
# モジュール読み込み 
import sys
sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin

#windowwidth = 320
#windowheight = 240
image_size = 300
nms_method = "hard"
label_path = "./models/forasp/open-images-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]


# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み 
net = IENetwork(model='./models/forasp/mbv3-ssd-cornv1.xml', weights='./models/forasp/mbv3-ssd-cornv1.bin')
exec_net = plugin.load(network=net)
input_blob_name = list(net.inputs.keys())[0]
output_blob_name = sorted(list(net.outputs.keys()))

#predictor
predictor = create_mobilenetv3_small_ssd_lite_predictor(exec_net, image_size = image_size,  nms_method=nms_method, input = input_blob_name, output = output_blob_name)

#print("stand", input_blob_name, output_blob_name)
# カメラ準備 
cap = cv2.VideoCapture(0)

if cap.isOpened() != True:
    print("camera open error!")
    quit()
else:
    print("camera open!")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)


def label_change(labels):
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = 2
        elif labels[i] == 2:
            labels[i] = 1

    return labels


# メインループ 
while True:
    ret, frame = cap.read()
    # Reload on error 
    if ret == False:
        continue
    frame = cv2.resize(frame, (300, 300))
    #print(frame.shape)
    boxes, labels, probs = predictor.predict(frame,10, 0.4) #TODO 閾値
    # 出力から必要なデータのみ取り出し 

    boxed = [] #重複box
    
    labels = label_change(labels)

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

        label = class_names[labels[i]] #+ str(probs[i])
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
    #key = cv2.waitKey(1)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
