import cv2

cap = cv2.VideoCapture("1615363610851.mp4")

# detecting moving objects from still frame
object_detector = cv2.createBackgroundSubtractorMOG2()

#Resizing frame's resolution and size
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:
    # Capturing frame-by-frame
    ret, frame = cap.read()

    #Detecting vehicles
    mask = object_detector.apply(frame)


    frame = rescale_frame(frame, percent=75)
    mask = rescale_frame(mask, percent=75)

    #Display resulting frame
    cv2.imshow("Orignal video 1615363610851", frame)
    #Display negative mask of resulting frame
    cv2.imshow("Mask", mask)


    # Press any key ---> speed play
    # Press q ---> QUIT
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
