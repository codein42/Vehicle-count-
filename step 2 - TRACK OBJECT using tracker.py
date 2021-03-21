import cv2
from tracker import *

#Create tracker object
tracker = EuclideanDistTracker()


cap = cv2.VideoCapture("Example.mp4")

# step 1: Detecting moving objects from still frame
object_detector = cv2.createBackgroundSubtractorMOG2(history=70, varThreshold= 300)

#Resizing frame's resolution and size
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:
    # Capturing frame-by-frame
    ret, frame = cap.read()
    height, weidth, _ = frame.shape
    print(height, weidth)

    #Defining woking area(region of interest{ROI})
    roi = frame[360: 720,320: 700]

    #Detecting vehicles
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    ##contour meaning defined boundary
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculating area and removing small elements
        area = cv2.contourArea(cnt)
        if area > 2000:
            #cv2.drawContours(roi, [cnt], -1, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])


    frame = rescale_frame(frame, percent=75)
    mask = rescale_frame(mask, percent=75)

    #step 2: object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.rectangle(roi, (x,y), (x + w, y + h), (255, 0, 0), 4)
    #print(boxes_ids)
    #Display resulting frame
    cv2.imshow("orignal VideoCapture", frame)
    #Display negative mask of resulting frame
    cv2.imshow("Mask", mask)
    #Display Region of working
    cv2.imshow("working place", roi)

    # Press any key ---> speed play
    # Press q ---> QUIT
    #if cv2.waitKey(0) & 0xFF == ord('q'):
    #    break
    key = cv2.waitKey(60)
    if key ==27:  # 27 is key code for "S"
        break
cap.release()
cv2.destroyAllWindows()
