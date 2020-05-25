# @aurthor: Akshat Surolia
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pageseg

coord=[]
def onMouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
       coord.append((x, y))
       print('x = %d, y = %d'%(x, y))

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


def PageCorrection(image):
	image=pageseg.main(image)
	height=image.shape[0]
	width=image.shape[1]
	cv.namedWindow('imaage', 0)
	cv.setMouseCallback('imaage', onMouse)
	cv.imshow("imaage",image)
	cv.waitKey(0)
	cv.destroyAllWindows()

	pts1 = np.float32([coord[0], coord[1], coord[2], coord[3]])
	pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
	matrix = cv.getPerspectiveTransform(pts1, pts2)
	result = cv.warpPerspective(image, matrix, (width, height))
	return result

path=os.getcwd()+'/Desktop/projectx/project'
filename= os.listdir(path)
imgcounter=1
for name in filename:
    
	filex= path+'/'+name
	print(filex)
	image = cv.imread(filename=filex)
	print(image.shape)
	
	result=PageCorrection(image)

	cv.imwrite(f'C:/Users/akroc/Desktop/result.png', result)
	print(result.shape)
	cropped= result
	
	cv.namedWindow('Perspective transformation', 0)
	cv.setMouseCallback('Perspective transformation', onMouse)
	cv.imshow("Perspective transformation", result)
	cv.waitKey(0)
	cv.destroyAllWindows()
	distance=coord[-1][1]-coord[-2][1]
	
	linepath='C:/Users/akroc/Desktop/lines'
	os.mkdir(f'{linepath}/Image{imgcounter}')
	linecounter=1
	for i in range(0,cropped.shape[0],distance):
		crop_img = cropped[i:i+distance, 0:cropped.shape[1]]
		cv.namedWindow("ion",0)
		cv.imshow("ion", crop_img)
		cv.waitKey(0)
		cv.imwrite(f'{linepath}/Image{imgcounter}/line{linecounter}.png', crop_img)

		lcropped = crop_img
		gray = cv.cvtColor(lcropped, cv.COLOR_BGR2GRAY)
		ret3, thresh2 = cv.threshold(
	    gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	
		cv.namedWindow('Orignal Image', 0)
		cv.imshow("Orignal Image", thresh2)
		cv.waitKey(0)

		thresh2 = 255-thresh2
		thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
		thresh2 = 255-thresh2

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

		thresh2 = np.transpose(thresh2)
		
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

		cv.namedWindow('newline', 0)
		cv.imshow("newline",thresh2)

		cv.waitKey(0)

		(components, _) = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

		wordcounter=1
		for c in components:
			
			# skip small word candidates
			if cv.contourArea(c) < 5000:
				continue
			
			currBox = cv.boundingRect(c) # returns (x, y, w, h)
			(x, y, w, h) = currBox
			currImg = lcropped[y:y+h, x:x+w]
			cv.namedWindow('finalword', 0)
			cv.imshow("finalword",currImg)
			cv.imwrite(f'C:/Users/akroc/Desktop/words/word{imgcounter}_{linecounter}_{wordcounter}.png', currImg)
			cv.waitKey(0)
			wordcounter+=1	

		cv.destroyAllWindows()
		linecounter+=1
	cv.destroyAllWindows()			
	imgcounter+=1


	