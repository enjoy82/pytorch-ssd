from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model_path = "./models/mbv3-Epoch-99-Loss-2.9210379077838016.pth" #モデル名
label_path = "./models/open-images-model-labels.txt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
net = net.to(DEVICE)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,  device=DEVICE)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box = box.numpy()
        print("trans", box, box[0], box[1], box[2], box[3])
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        cv2.putText(orig_image, label,
                    (int(box[0])+20, int(box[1])+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    5,
                    cv2.LINE_AA)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
