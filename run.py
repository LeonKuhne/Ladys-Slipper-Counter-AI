from darkflow.net.build import TFNet
import cv2
import numpy as np

image_name = 'DJI_0070.JPG'


def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    count = 0
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        if confidence > 0.4:
            label = f"{result['label']}[{count}] {str(round(confidence, 3))}"
            count += 1
            newImage = cv2.rectangle(
                newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5),
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return count, newImage


options = {"model": "cfg/yolo_ls.cfg",
           "batch": 8,
           "epoch": 1000,
           "gpu": 1.0,
           "train": True,
           "annotation": "./annotations/",
           "dataset": "./images/",
           "load": -1}

tfnet = TFNet(options)
imgcv = cv2.imread(f"./sample_img/{image_name}")
results = tfnet.return_predict(imgcv)
count, newImg = boxing(imgcv, results)
cv2.imwrite('./prediction.png', newImg)

print("number of ladys slippers:", count)
