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




	cropped= result
	
	cv.namedWindow('Perspective transformation', 0)
	cv.setMouseCallback('Perspective transformation', onMouse)
	cv.imshow("Perspective transformation", result)
	cv.waitKey(0)
    
    lcropped= result
    blur = cv.GaussianBlur(lcropped, (7, 7), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    thresh2 = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)

    cv.namedWindow('Orignal Image', 0)
    cv.imshow("Orignal Image", thresh2)
    cv.waitKey(0)

    thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))


    # Specify size on vertical axis
    rows = thresh2.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 19))
    # Apply morphology operations
    vertical = cv.erode(thresh2, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    cv.namedWindow("LineVertical",0)
    cv.imshow('LineVertical', vertical)
    # cv.show_wait_destroy("vertical", vertical)
    thresh2=vertical

    cv.namedWindow('Lines Removed', 0)
    cv.imshow("Lines Removed", thresh2)
    cv.waitKey(0)

    thresh2 = 255-thresh2
    thresh2 = cv.medianBlur(thresh2, 3)

    cv.namedWindow('MedianBlur', 0)
    cv.imshow("MedianBlur", thresh2)
    cv.waitKey(0)
    cv.destroyAllWindows()


    thresh2 = 255-thresh2
    thresh2 = cv.dilate(thresh2, np.ones(25))


    thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=3)

    rgb_img = cv.cvtColor(thresh2, cv.COLOR_GRAY2BGR)

    cv.namedWindow('TXTX', 0)
    cv.imshow('TXTX', rgb_img)
    cv.waitKey(0)


    def countfreq(list):
        white=np.count_nonzero(list)
        return (white,len(list)-white)

    def isLine(list):
        size= len(list)
        white,black = countfreq(list)
        ratio=(white/size,black/size)
        # print(white,black,ratio)
        if(ratio[1]>0.97):
            return 1
        else:
            return 0


    height=thresh2.shape[0]
    width=thresh2.shape[1]
    # print(height,width)
    # for i in range(height):
    #     print("\n-------------------------------------------------------------------------")
    #     for j in range(int(width/2)):
    #         print(thresh2[i][j],end=" ")

    lineornot=[]
    for i in range(height):
        lineornot.append(isLine(thresh2[i]))

    def rectify(list):
        for i in range(3,len(list)-3,3):
            if(np.count_nonzero(list[i-3:i+3])<3):
                list[i-3:i+3]=[0]*6
        return list
        
    lineornot=rectify(lineornot)


    for i in range(height):
        if(lineornot[i]==True):
            thresh2[i]=[255]*width

    cv.namedWindow('d', 0)
    cv.imshow("d",thresh2)
    cv.waitKey(0)


    thresh2=255-thresh2
    (components, _) = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    line1=1
    for c in components:
        
        # skip small word candidates
        if cv.contourArea(c) < 150000:
            continue
        
        currBox = cv.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = lcropped[y-10:y+h+10, 0:thresh2.shape[1]]
        cv.namedWindow('finalword', 0)
        cv.imshow("finalword",currImg)
        cv.imwrite(f'C:/Users/akroc/Desktop/line{line1}.png', currImg)
        cv.waitKey(0)
        line1+=1
    cv.destroyAllWindows()






    lcropped = cv.imread("C:/Users/akroc/Desktop/line4.png")
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


    def isWord(list):
        size = len(list)
        white, black = countfreq(list)
        ratio = (white/size, black/size)
        if(ratio[0] > 0.945
        ):
            return 1
        else:
            return 0


    wordornot = []
    for i in range(height):
        wordornot.append(isWord(thresh2[i]))



    def rectifyw(list):
        for i in range(30, len(list)-30, 30):
            if(np.count_nonzero(list[i-30:i+30]) < 30):
                list[i-30:i+30] = [0]*60
        return list


    wordornot = rectifyw(wordornot)


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
        cv.waitKey(0)
        wordcounter+=1	

    cv.destroyAllWindows()
