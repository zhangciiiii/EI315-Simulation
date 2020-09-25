import numpy as np
import cv2
from collections import Counter
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
    # ------use a dictionary to store info -------
    if state.get() is None:
        info = {"step":0, "sign":35, "sign_flags":[], "sign_counters":[], "current_tempt":0}
        state.set(info)
    else:
        info = state.get()
       
    # imwrite(str(state.get()) + '-1.jpg', view1)
    # imwrite(str(state.get()) + '-2.jpg', view2)

    info["step"]+=1   
    log.append("# Step:"+str(info["step"]) )

    #----------sign detection---------------------
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
        
        id_num = 0
        if rect:
            xmin, ymin, xmax, ymax = rect
            sign_flag = 1
            if xmax > 500:
                roi = im[ymin:ymax, xmin:xmax, :]
                id_num = svm.predict(roi, "hog")

                if info["current_tempt"]==0:
                    info["current_tempt"] = id_num
                    log.append('current tempt:'+str(info["current_tempt"]))

                elif info["current_tempt"]!=id_num:
                    id_num = info["current_tempt"]
                    log.append('switch:'+str(id_num) + ' to '+str(info["current_tempt"]))
                
                #log.append("id:" + str(id_num))
                log.append(sign_classes[id_num])
                
                

        else:
            sign_flag = 0
    #--------------------------------------------


    #----------update the info-------------------
    if len(info["sign_flags"]) >= 20:
        info["sign_flags"].pop(0)
        info["sign_flags"].append(sign_flag)
    else:
        info["sign_flags"].append(sign_flag)

    if len(info["sign_counters"]) >= 30:
        info["sign_counters"].pop(0)
        info["sign_counters"].append(id_num)
    else:
        info["sign_counters"].append(id_num)
    
    
    state.set(info)
    
    #-------------------------------------------


    # -----------direction control-----------------

    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    lower_white = np.array([0,0,221])
    upper_white = np.array([180,30,255])

    hsv = cv2.cvtColor(view1, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)


    '''normal'''
    if  sum(info["sign_flags"]) == 0:
        info["current_tempt"]=0
        log.append('current tempt:'+str(info["current_tempt"]))

    if mask_white[0:60,40:120].sum() >10:
        left_speed = 0
        right_speed = 0.5

    elif mask_yellow[0:60,40:120].sum() >10:
        left_speed = 0.5
        right_speed = 0

    else :
        left_speed = right_speed = 0.5
    
    ''' meet the sign'''
    if  sum(info["sign_flags"]) != 0:
        if mask_yellow[0:60,40:120].sum() >10:
            left_speed = 0.1
            right_speed = 0
        else:
            left_speed = right_speed = 0.5

    '''sign: straight'''
    if 35 in info["sign_counters"][0:3]:        
        left_speed = 0.5
        right_speed = 0.5

    '''sign: right'''
    if 33 in info["sign_counters"][0:3]:        
        left_speed = 0.5
        right_speed = 0.0

    '''sign: left'''
    if 34 in info["sign_counters"][0:3]:        
        left_speed = 0.0
        right_speed = 0.2

    #----------------------------------------------

    #imwrite(str(info["step"]) + '.jpg', mask_white[60:,40:120])
    # log.append(str(mask_white[60:,40:120].sum()))

    return left_speed, right_speed
