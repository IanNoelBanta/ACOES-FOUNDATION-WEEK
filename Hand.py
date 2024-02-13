import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import random

cap = cv2.VideoCapture(0)

heartPNG = cv2.imread("heart2.png",cv2.IMREAD_UNCHANGED)
timer = 0

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

while True:
    if timer == 0:
        dim1 = (random.randrange(0,481), random.randrange(0,640))
        dim2 = (random.randrange(0,481), random.randrange(0,640))
        timer = 3

    success, img = cap.read()
    print(img.shape)

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, draw=False, flipType=False)

    if hands:
        hand1 = hands[0]  
        lmList1 = hand1["lmList"]  
        bbox1 = hand1["bbox"]  
        center1 = hand1['center']  
        handType1 = hand1["type"]  

        fingers1 = detector.fingersUp(hand1)

        length, info, img = detector.findDistance(lmList1[3][0:2], lmList1[7][0:2], img, color=(255, 0, 255),
                                            scale=10)
        
        if length <= 20:
            cv2.putText(img, "I     CPE!!", (lmList1[3][0]-50, lmList1[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(img, "I     YOU!!", (lmList2[3][0]-50, lmList2[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            img = cvzone.overlayPNG(img, heartPNG, pos=dim1)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            length, info, img = detector.findDistance(lmList2[3][0:2], lmList2[7][0:2], img, color=(255, 0, 0),
                                                      scale=10)
            
            if length <= 20:
                cv2.putText(img, "I     ENGINEERING!!", (lmList2[3][0]-50, lmList2[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(img, "I     YOU!!", (lmList2[3][0]-50, lmList2[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                img = cvzone.overlayPNG(img, heartPNG, pos=dim2)


        print(" ")

    timer -= 1

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()