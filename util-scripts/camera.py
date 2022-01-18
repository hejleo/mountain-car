# Script for images acquisition from the webcam

import cv2

vid = cv2.VideoCapture(0)

cont = 0

while(True):

    cv2.waitKey(50)
    ret, frame = vid.read()

    # Display the resulting frame
    img = cv2.imshow('frame', frame)
    if ret:
        cv2.imwrite("/home/pulp/openai-gym/dataset/test/right-"+str(cont)+".jpg", frame)
        cont += 1

vid.release()
cv2.destroyAllWindows()
