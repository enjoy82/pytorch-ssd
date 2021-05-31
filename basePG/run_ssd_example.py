from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys


image_path = "./images/test/0b9edee2d367afd3.jpg"

model_path = "./models/mbv3-v2-Epoch-65-Loss-4.263951171823099.pth" #モデル名
label_path = "./models/open-images-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "./images/test_ball_out.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
