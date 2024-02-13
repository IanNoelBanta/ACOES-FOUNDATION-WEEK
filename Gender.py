import cv2
import math
import time
import argparse
import random

def getFaceBox(net, frame,conf_threshold = 0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]* frameWidth)
            y1 = int(detections[0,0,i,4]* frameHeight)
            x2 = int(detections[0,0,i,5]* frameWidth)
            y2 = int(detections[0,0,i,6]* frameHeight)
            bboxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)

    return frameOpencvDnn , bboxes

faceProto = "weights/opencv_face_detector.pbtxt"
faceModel = "weights/opencv_face_detector_uint8.pb"

ageProto = "weights/age_deploy.prototxt"
ageModel = "weights/age_net.caffemodel"

genderProto = "weights/gender_deploy.prototxt"
genderModel = "weights/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
pangetList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-80)', '(80-95)', '(95-98)', '(99)']
puriList = ['ANG GANDA MO!!', 'ANG POGI MO!!']

ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(0)
padding = 20
timer = 0

while cv2.waitKey(1) < 0:
    if timer == 0:
        panget_rate = random.randrange(0,11)
        timer = 100

    #read frame
    t = time.time()
    hasFrame , frame = cap.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frame = cv2.flip(frame,1)   
    # small_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)
    small_frame = cv2.resize(frame,(640, 480),fx = 0.5,fy = 0.5)

    frameFace ,bboxes = getFaceBox(faceNet,small_frame)
    # if not bboxes:
    #     # print("No face Detected, Checking next frame")
    #     continue
    for bbox in bboxes:
        face = small_frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        if gender == 'Male':
            puri = puriList[1]
        else:
            puri = puriList[0]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label1 = "Gender:{}".format(gender)
        label2 = "Age:{}".format(age)
        label3 = "Rating:{}".format(pangetList[panget_rate])
        label4 = "Klaymeyt, {}".format(puri)
        cv2.putText(frameFace, label1, (bbox[0], bbox[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frameFace, label2, (bbox[0], bbox[1]-75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frameFace, label3, (bbox[0], bbox[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frameFace, label4, (bbox[0], bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Age Gender Demo", frameFace)
    
    # print(timer)
    timer -= 1
       
    # print("time : {:.3f}".format(time.time() - t))


    




        
        