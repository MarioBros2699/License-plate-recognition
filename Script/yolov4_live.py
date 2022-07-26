import cv2 as cv
import time
import numpy as np
import easyocr

def get_roi(frame, box):
    left_x, top_y, w, h = box
    roi = frame[top_y:top_y+h,left_x:left_x+w]
    return roi;

def omografia (roi, modello):
    model_gray = cv.cvtColor(modello, cv.COLOR_BGR2GRAY)
    targa_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    model_gray = cv.resize(model_gray, (targa_gray.shape[1],targa_gray.shape[0]), interpolation = cv.INTER_AREA)  
    sift = cv.xfeatures2d.SIFT_create()
    kp_model, dec_model = sift.detectAndCompute(model_gray, None)
    kp_targa, des_targa = sift.detectAndCompute(targa_gray, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dec_model, des_targa, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.7* n.distance:
            good_points.append(m)
    if len(good_points) > 3:
        query_pts = np.float32([kp_model[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_targa[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(train_pts, query_pts, cv.RANSAC, 5.0)
        h, w, channels = modello.shape
        omo = cv.warpPerspective(roi, matrix, (w,h))
        return omo

def get_targa(frame, box, modello):
    reader = easyocr.Reader(['en'])
    roi = get_roi(frame, box)
    result = reader.readtext(roi)
    text1 = None
    text = ''
    #gestisco omografina nera
    try:
        omo = omografia(roi, modello)
        result1 = reader.readtext(omo)
        for detection in result1: 
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text1 = detection[1]
    except:
        text1 = ''
    for detection in result: 
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
    if len(text1) >= len(text):
        return text1
    else:
        return text

Conf_threshold = 0.6
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
net = cv.dnn.readNet('yolov4.weights', 'yolov4-custom.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

modello = cv.imread("modello.jpg")
cap = cv.VideoCapture(0)
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

dim = (int(frame_width), int(frame_height))
print(dim)
starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()

    frame_counter += 1
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        text = get_targa(frame, box, modello)
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f targa: %s" % (class_name[classid], score, text)
        cv.rectangle(frame, box, color, 1)
        # cv.line(frame, (box[0]-3, box[1]-15),
        #         (box[0]+110, box[1]-15), (0, 0, 0), 15)
        cv.rectangle(frame, (box[0]-2, box[1]-20),
                     (box[0]+230, box[1]-4), (100, 130, 100), -1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.line(frame, (18, 43), (140, 43), (0, 0, 0), 27)
    cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    cv.imshow('frame', frame)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()

