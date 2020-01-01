import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread("book2.jpg")
orb = cv2.ORB_create(nfeatures=50)
keypoints_img, descriptors_img = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while(True):
    ret, frame = cap.read()
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)
    matches = bf.match(descriptors_img, descriptors_frame)
    matches = sorted(matches, key = lambda x:x.distance)
    matching_result = cv2.drawMatches(img, keypoints_img, frame, keypoints_frame, matches[:20], None)
    cv2.imshow("Matching result", matching_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
