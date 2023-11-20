#############################################################################
#                                                                           #
# This is the testing part of the Tone Modifier.                            #
# Xiao-Yong Wei built for COMP 4423                                         #
#                                                                           #
#############################################################################

import cv2,numpy as np,copy,random,time

########################### Keypoint Sampling ############################
# declare a video capture object
vid = cv2.VideoCapture('pexels-wong-lun-6112093.mp4')
#vid = cv2.VideoCapture('campus.mp4')
inversion_on=False
keypoints_on=False

sift = cv2.SIFT_create()
detector=sift
tones=np.load('tones.npy')

def change_tone(frame,shift):
    # convert the image data to HSV space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    # modify hue channel by adding shift and modulo 180
    h2 = np.mod(h*0.0 + shift, 180).astype(np.uint8)
    # convert back to RGB space
    frame_new = cv2.cvtColor(cv2.merge([h2,s,v]), cv2.COLOR_HSV2BGR)
    return frame_new

vid_writer = None
icounter=0

vid_writer = cv2.VideoWriter('tone.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (int(vid.get(3)),int(vid.get(4))))

while vid.isOpened():
    # capture the video frame by frame
    ret, frame = vid.read()
    if not ret: break
    
    # if not icounter%20==0: # speed-up by subsampling
    #     continue
    
    #frame = cv2.flip(frame, 1) # optional
    # get dimensions of the frame
    h, w, c = frame.shape
    print('count=',icounter,h,w,c)
    
    
    
    # #extract key points
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # kp, des = detector.detectAndCompute(gray, None)
    # #print(type(kp))
    # for feat in des:
    #     features.append(feat.flatten())
        
    # keyframe_list.append(copy.deepcopy(des))
    
    frame=change_tone(frame,tones[icounter])
    # display the resulting frame
    cv2.imshow('COMP 4423', 255-frame if inversion_on else frame)
        
    vid_writer.write(frame)
    
    icounter+=1
    
    key=cv2.waitKey(1) & 0xFF
    # quit when 'q' is pressed
    if  key== ord('q'):
        break
    if key==ord('i'):
        inversion_on=not inversion_on
    if key==ord('k'):
        keypoints_on=not keypoints_on
    
# release the cap object
vid.release()
# destroy all the windows
cv2.destroyAllWindows()
vid_writer.release()

print('Done!')

