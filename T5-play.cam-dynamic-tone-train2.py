#############################################################################
#                                                                           #
# This is the training part of the Tone Modifier.                           #
# Xiao-Yong Wei built for COMP 4423                                         #
#                                                                           #
#############################################################################

import cv2,numpy as np,copy

########################### Keypoint Sampling ############################
# declare a video capture object
vid = cv2.VideoCapture('pexels-wong-lun-6112093.mp4')
#vid = cv2.VideoCapture('campus.mp4')
inversion_on=False
keypoints_on=False

sift = cv2.SIFT_create()
detector=sift

icounter=0
features=[]
keyframe_list=[]
Subsample_rate=20
while vid.isOpened():
    # capture the video frame by frame
    ret, frame_raw = vid.read()
    if not ret: break
    icounter+=1
    if not icounter%Subsample_rate==0: # speed-up by subsampling
        continue
    
    frame = cv2.flip(frame_raw, 1) # optional
    # get dimensions of the frame
    h, w, c = frame.shape
    print('count=',icounter,h,w,c)
    
    #extract key points
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    #print(type(kp))
    for feat in des:
        features.append(feat.flatten())
        
    keyframe_list.append(copy.deepcopy(des))
      
    # display the resulting frame
    #cv2.imshow('COMP 4423', 255-frame if inversion_on else frame)
    
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

print('len(features)=',len(features))


########################### Vocabulary Learning ############################
from os import path

num_clusters=128 #this is also the nubmer of words in the vocabulary
if path.exists('voc.npy'):
    centers=np.load('voc.npy')
    num_clusters=len(centers)
else:
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(np.stack(features, axis=0)), num_clusters, None, criteria, 10, flags)
    np.save('voc.npy',centers)

########################### Vocabulary Learning ############################

# generate BoVW based on the vocabulary (i.e., centers)
from scipy.cluster.vq import *
from sklearn import preprocessing

def des_to_BoVG(des):
    bog_feat=np.zeros(num_clusters,"float32")
    words, distance = vq(des,centers) # map the descriptor into words
    # count the frequency of words
    for w in words:
        bog_feat[w] += 1
    return bog_feat

kf_bags = np.zeros((len(keyframe_list), num_clusters), "float32")
for i in range(len(keyframe_list)):
    kf_bags[i]=des_to_BoVG(keyframe_list[i])
#print(kf_bags)
kf_bags = preprocessing.normalize(kf_bags, norm='l2')
np.save('keyframe_feats.npy',kf_bags)

########################### Tone Assignment ############################
import random
from numpy.linalg import norm
# option 1: assign a random hue to each keyframe
#key_tones=[random.randint(0,180) for i in range(len(kf_bags))]
# option 2: assign tones to keyframes with a gradual change
key_tones=[i*(180/len(kf_bags)) for i in range(len(kf_bags))]
print('keytones',key_tones)

vid = cv2.VideoCapture('pexels-wong-lun-6112093.mp4')
icounter=0
tones=[]
tone_by_similarity=True
while vid.isOpened():
    # capture the video frame by frame
    ret, frame = vid.read()
    if not ret: break
    
    print('processing on frame ',str(icounter)) 
    
    if tone_by_similarity:
        #extract key points
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        bog_feat=des_to_BoVG(des)
        
        #compare to key frame features
        scores = np.dot(bog_feat, kf_bags.T)
        #print(scores)
        ranking = np.argsort(-scores)
        #print(ranking[0:3])
        
        start_tone=key_tones[ranking[0]] 
        end_tone=key_tones[ranking[1]] 
        weight=np.dot(bog_feat, kf_bags[ranking[0]])/(norm(bog_feat)*norm(kf_bags[ranking[0]]))
    else:
        neigbhour_key_ind=icounter//Subsample_rate 
        neigbhour_key_ind=neigbhour_key_ind if neigbhour_key_ind<len(key_tones) else len(key_tones)-1
        start_tone=key_tones[neigbhour_key_ind] 
        end_tone=start_tone if neigbhour_key_ind+1>=len(key_tones) else key_tones[neigbhour_key_ind+1]
        weight=1.0-(icounter%Subsample_rate)/Subsample_rate
    
    tone=int(start_tone*weight+end_tone*(1.0-weight))
    #print(icounter,neigbhour_key_ind,start_tone,end_tone,tone)
    tones.append(tone)
    icounter+=1

# release the cap object
vid.release()
# destroy all the windows
cv2.destroyAllWindows()

print('Tone assignement done!') 
np.save('tones.npy',np.array(tones))
    
