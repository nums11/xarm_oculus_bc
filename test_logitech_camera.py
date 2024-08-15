import os
import numpy as np
import cv2
import time

# define a video capture object
vid = cv2.VideoCapture(0 + cv2.CAP_V4L2)
time.sleep(1)

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    resized = cv2.resize(frame, (512, 512))
    cv2.imwrite('testimage.jpg', frame)
    # predictImageClass(resized)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     # resized = cv2.resize(frame, (256, 256))
    #     # cv2.imwrite('testimage.jpg', frame)
    #     # predictImageClass(resized)
    #     break
    # if i == 15:
    #     cv2.imwrite('testimage.jpg', resized)
    #     break
    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()