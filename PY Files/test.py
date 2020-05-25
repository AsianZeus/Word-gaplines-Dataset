import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import math


def countfreq(list):
    white = np.count_nonzero(list)
    return (white, len(list)-white)


def isWord(list):
    size = len(list)
    white, black = countfreq(list)
    ratio = (white/size, black/size)
    if(ratio[0] > 0.945
	):
        return 1
    else:
        return 0


def rectify(list):
    for i in range(30, len(list)-30, 30):
        if(np.count_nonzero(list[i-30:i+30]) < 30):
            list[i-30:i+30] = [0]*60
    return list


pathx = 'C:/Users/akroc/Desktop/lines/Image1'
filename = os.listdir(pathx)
for name in filename:
	newpathx = pathx+'/'+name
	lcropped = cv.imread(newpathx)
	gray = cv.cvtColor(lcropped, cv.COLOR_BGR2GRAY)
	
	ret3, thresh2 = cv.threshold(
	    gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	
	cv.namedWindow('Orignal Image', 0)
	cv.imshow("Orignal Image", thresh2)
	cv.waitKey(0)

	thresh2 = 255-thresh2
	thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
	thresh2 = 255-thresh2

	# cv.namedWindow('After Dialation 1', 0)
	# cv.imshow("After Dialation 1", thresh2)
	# cv.waitKey(0)

	laplacian = cv.Laplacian(gray, -1, ksize=15)
	minLineLength = 60
	maxLineGap = 60
	lines = cv.HoughLinesP(laplacian,5,np.pi/180,550,minLineLength,maxLineGap)
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv.line(thresh2,(x1,y1),(x2,y2),(255,255,255),9)

	cv.namedWindow('Lines Removed', 0)
	cv.imshow("Lines Removed", thresh2)
	cv.waitKey(0)

	thresh2 = cv.medianBlur(thresh2, 3)
	
	cv.namedWindow('MedianBlur', 0)
	cv.imshow("MedianBlur", thresh2)
	cv.waitKey(0)

	thresh2 = 255-thresh2
	thresh2 = cv.dilate(thresh2, np.ones(5))
	thresh2 = 255-thresh2


	# print(thresh2.shape)
	thresh2 = np.transpose(thresh2)
	# print(thresh2.shape)


	height = thresh2.shape[0]
	width = thresh2.shape[1]

	wordornot = []
	for i in range(height):
		wordornot.append(isWord(thresh2[i]))


	wordornot = rectify(wordornot)

	# plt.barh(range(0, height), wordornot, align='center', alpha=0.5)
	# plt.xlabel('word')
	# plt.title('word Or Not')
	# plt.show()


	for i in range(height):
		if(wordornot[i]==True):
			thresh2[i]=[0]*width

	thresh2 = np.transpose(thresh2)

	cv.namedWindow('d', 0)
	cv.imshow("d",thresh2)

	cv.waitKey(0)

	(components, _) = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	linecounter=1
	for c in components:
    	
		# skip small word candidates
		if cv.contourArea(c) < 6000:
			continue
		# append bounding box and image of word to result list
		currBox = cv.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = lcropped[y:y+h, x:x+w]
		cv.namedWindow('d', 0)
		cv.imshow("d",currImg)
		# cv.imwrite(f'C:/Users/akroc/Desktop/words/word_{linecounter}.png', currImg)
		cv.waitKey(0)
		linecounter+=1		
		cv.destroyAllWindows()