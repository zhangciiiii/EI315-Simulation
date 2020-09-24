import numpy as np
import cv2
from services.detection import detection
from services.svm import SVM
from services.utils import imwrite

# if you want print some log when your program is running, 
# just append a string to this variable
log = []


def image_to_speed(view1, view2, state):
    """This is the function where you should write your code to 
    control your car.
    
    You need to calculate your car wheels' speed based on the views.
    
    Whenever you need to print something, use log.append().
  
    Args:
        view1 (ndarray): The left-bottom view, 
                          it is grayscale, 1 * 120 * 160
        view2 (ndarray): The right-bottom view, 
                          it is colorful, 3 * 120 * 160
        state: your car's state, initially None, you can 
               use it by state.set(value) and state.get().
               It will persist during continuous calls of
               image_to_speed. It will not be reset to None 
               once you have set it.
  
    Returns:
        (left, right): your car wheels' speed
    """
    if state.get() is None:
        state.set(1)
    else:
        state.set(state.get() + 1)
    log.append("#" + str(state.get()))
    imwrite(str(state.get()) + '-1.jpg', view1)
    imwrite(str(state.get()) + '-2.jpg', view2)


    left_speed = right_speed = 1


    if view2 is not None:
        sign_classes = {
            14: 'Stop',
            33: 'Turn right',
            34: 'Turn left',
            35: 'Straight'
        }
        svm = SVM()
        detector = detection()
        im = view2
        rect = detector.ensemble(im)
        if rect:
            xmin, ymin, xmax, ymax = rect
            if xmax < 600:
                roi = im[ymin:ymax, xmin:xmax, :]
                id_num = svm.predict(roi, "hog")
                sign_flag = 1
                log.append("id:" + str(id_num))
                log.append(sign_classes[id_num])

    return left_speed, right_speed
