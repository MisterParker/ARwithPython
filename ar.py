import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# set the target image
targetImg = cv2.imread('gerua_scaled.jpg')
targetImg = cv2.resize(targetImg, (400, 400))
h,w,c = targetImg.shape

inputVid = cv2.VideoCapture('batting.mp4')

#using ord detector for feature extraction
orb = cv2.ORB_create(nfeatures=1000)
keyPoint1, desc1 = orb.detectAndCompute(targetImg, None)
targetImg = cv2.drawKeypoints(targetImg, keyPoint1, None)

success, webCamImg = cam.read()
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('AR_feature_out.mp4', fourcc, 10.0, (webCamImg.shape[1], webCamImg.shape[0])) 
# out = cv2.VideoWriter('AR_feature_out.mp4', fourcc, 10.0, (1680, 720)) 

while True:
	success, webCamImg = cam.read()
	imgAug = webCamImg.copy()
	keyPoint2, desc2 = orb.detectAndCompute(webCamImg, None)
	webCamImg = cv2.drawKeypoints(webCamImg, keyPoint2, None)
	# cv2.imshow("webCamVid", webCamImg)

	ret, vidImg = inputVid.read()
	#resizing to fit to the target image
	vidImg = cv2.resize(vidImg, (h, w))

	brute_force = cv2.BFMatcher()
	matches = brute_force.knnMatch(desc1, desc2, k=2)
	good_matches = []
	for m1,m2 in matches:
		if m1.distance < 0.65*m2.distance:
			good_matches.append(m1)

	# print(len(good_matches))
	imgFeatures = cv2.drawMatches(targetImg, keyPoint1, webCamImg, keyPoint2, good_matches, None, flags = 2)

	# find homogenity
	if len(good_matches) > 10:
		srcPts = np.float32([keyPoint1[m1.queryIdx].pt for m1 in good_matches]).reshape(-1,1,2)
		dstPts = np.float32([keyPoint2[m1.trainIdx].pt for m1 in good_matches]).reshape(-1,1,2)

		matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
		print(matrix)
		
		#draw bb
		pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts, matrix)
		img2 = cv2.polylines(webCamImg, [np.int32(dst)], True, (255, 0, 0), 2)

		imgWarp = cv2.warpPerspective(vidImg, matrix, (webCamImg.shape[1], webCamImg.shape[0]))

		# creating overlay
		maskNew = np.zeros((webCamImg.shape[0], webCamImg.shape[1]), np.uint8) 
		cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255)) # this will create a mask with white overlay, 
															  # but we need to inverse so that we can add the mask with actual image
		# cv2.imshow("maskNew", maskNew) 
		# invert the mask
		maskInverse = cv2.bitwise_not(maskNew)
		# cv2.imshow("maskInverse", maskInverse) 

		# now augment the background 
		imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInverse)

		# cv2.imshow("imgAug", imgAug)
		# cv2.imshow("imgbb", img2)
		
		# cv2.imshow("imgWarp", imgWarp)

		imgAR = cv2.bitwise_or(imgAug, imgWarp)
		# cv2.imshow("img AR", imgAR)

		# out.write(imgAR)

	cv2.imshow("imgFeature", imgFeatures)
	print(imgFeatures.shape)
	# out.write(imgFeatures)
	# cv2.imshow("inputVid", vidImg)
	# cv2.imshow("targetImg", targetImg)

	if cv2.waitKey(10) & 0xFF == ord('q'): 
		cam.release()
		#out.release()
		cv2.destroyAllWindows()
		break
