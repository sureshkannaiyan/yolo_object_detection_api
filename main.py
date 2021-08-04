# YOLO object detection
import os
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify

UPLOAD_FOLDER = "./upload/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def yolo_object_detection(img):
    # Load names of classes
    classes = open('coco.names').read().strip().split('\n')

    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    all_detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            all_detected_objects.append({"object": classes[classIDs[i]],
                                         "object_location": (x, y, w, h), "confidence_score": confidences[i]})

    return jsonify(all_detected_objects)


@app.route("/inputimage", methods=['GET', 'POST'])
def detect_object():
    if request.method == 'POST':
        imagefile = request.files["image"]
        if imagefile:
            path = os.path.join(app.config['UPLOAD_FOLDER'], "img.png")
            imagefile.save(path)
            image = cv.imread(path)
            obj_det = yolo_object_detection(image)
        else:
            obj_det = "image not received"
    else:
        obj_det = "Request could not be processed"
    return obj_det

if __name__ == "__main__":
    app.run(debug=False)

