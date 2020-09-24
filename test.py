import numpy as np
import cv2

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img


view1 = cv_imread("3-1.jpg") # yellow_white


lower_yellow = np.array([26, 43, 46])
upper_yellow = np.array([34, 255, 255])
lower_white = np.array([0,0,221])
upper_white = np.array([180,30,255])

hsv = cv2.cvtColor(view1, cv2.COLOR_BGR2HSV)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

print(mask_white.sum())

'''
cv2.imwrite( "yellow_mask.jpg" , mask_yellow)
cv2.imwrite( "white_mask.jpg" , mask_white)
'''