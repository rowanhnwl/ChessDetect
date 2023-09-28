import cv2 as cv
from mss import mss
import numpy as np

bounding_box = {'top': 293, 'left': 64, 'width': 658, 'height': 658}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    cv.imshow('screen', np.array(sct_img))

    if (cv.waitKey(1) & 0xFF) == ord('q'):
        cv.destroyAllWindows()
        break