import cv2
import numpy as np

MIN_MATCH_COUNT = 10

cap = cv2.VideoCapture(0)
img = cv2.imread("book2.jpg")
orb = cv2.ORB_create(nfeatures=1000)
keypoints_img, descriptors_img = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while(True):
    ret, frame = cap.read()
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)
    matches = bf.match(descriptors_img, descriptors_frame)
    matches = sorted(matches, key = lambda x:x.distance)

    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints_img[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 9.0)
        h, w, _ = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M) 
        frame = cv2.polylines(frame, [np.int32(dst)], True,(0,255,0),4)

    matching_result = cv2.drawMatches(img, keypoints_img, frame, keypoints_frame, matches[:50], 0, matchColor = (0,255,0), flags=2)
    cv2.imshow("Matching result", matching_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
