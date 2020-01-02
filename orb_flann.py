import cv2
import numpy as np

MIN_MATCH_COUNT = 10
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)

cap = cv2.VideoCapture(0)
img1 = cv2.imread("book2.jpg")
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
flann = cv2.FlannBasedMatcher(index_params, {})

while(True):
    ret, frame = cap.read()
    kp2, des2 = orb.detectAndCompute(frame, None)    
    matches = flann.knnMatch(des1,des2,k=2)  
    
    good = []
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7*n.distance:
                good.append(m)
        except ValueError:
            pass
   
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 9.0)
        matchesMask = mask.ravel().tolist()
        h,w, _ = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        frame = cv2.polylines(frame,[np.int32(dst)],True,(0,255,0),4)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    img3 = cv2.drawMatches(img1,kp1,frame,kp2,good,0,**draw_params)
    cv2.imshow('Matching result', img3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
