import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import CharSegmentation
import PageSegmentation


path=os.getcwd()+'/Desktop/Dataset'
try:
    cseg=CharSegmentation.Model('C:/Users/akroc/Desktop/ocr-handwriting-models/gap-clas/RNN/Bi-RNN-new', 'prediction')
    print("Successfully loaded.")
except:
    print("Couldn't Load the model!")

coord=[]

def countfreq(list):
    white=np.count_nonzero(list)
    return (white,len(list)-white)

def onMouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
       coord.append((x, y))
       print('x = %d, y = %d'%(x, y))

# def PageCorrection(image):
    
#     globals()['coord']=[]
#     image=PageSegmentation.main(image)
#     height=image.shape[0]
#     width=image.shape[1]
#     cv.namedWindow('Page Segmented Image', 0)
#     cv.setMouseCallback('Page Segmented Image', onMouse)
#     cv.imshow("Page Segmented Image",image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

#     pts1 = np.float32([coord[0], coord[1], coord[2], coord[3]])
#     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#     print("Points:  ",pts1,pts2)
#     matrix = cv.getPerspectiveTransform(pts1, pts2)
#     result = cv.warpPerspective(image, matrix, (width, height))
#     return result


# filename= os.listdir(path+'/Images')
# imgcounter=25

# for name in filename:
#     file=path+'/Images/'+name
#     print(file)
#     image=cv.imread(file)

#     #****************** Page Frame Segmentation **********************
#     PageSegmentedImage =PageCorrection(image)
    
#     cv.imwrite(f'{path}/CroppedImage/Image{imgcounter}.png', PageSegmentedImage)
#     print(f"Image Shape{PageSegmentedImage.shape}")

#     #************ Four Point Perspective *******************
#     cv.namedWindow("Perspective Trasform",0)
#     cv.setMouseCallback("Perspective Trasform",onMouse)
#     cv.imshow("Perspective Trasform",PageSegmentedImage)
#     cv.waitKey(0)

#     blur = cv.GaussianBlur(PageSegmentedImage, (7, 7), 0)
#     gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
#     thresh2 = cv.adaptiveThreshold(
#         gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)

#     # cv.namedWindow('Orignal Image', 0)
#     # cv.imshow("Orignal Image", thresh2)
#     # cv.waitKey(0)

#     #***************************Line Removal****************************
#     thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))

#     # Specify size on vertical axis
#     rows = thresh2.shape[0]
#     # Create structure element for extracting vertical lines through morphology operations
#     verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 20))
#     # Apply morphology operations
#     thresh2 = cv.erode(thresh2, verticalStructure)
#     thresh2 = cv.dilate(thresh2, verticalStructure)
    
#     # Show extracted vertical lines
#     # cv.namedWindow('Lines Removed Image', 0)
#     # cv.imshow("Lines Removed Image", thresh2)
#     # cv.waitKey(0)

#     thresh2 = 255-thresh2
#     thresh2 = cv.medianBlur(thresh2, 3)

#     # cv.namedWindow('MedianBlur Image', 0)
#     # cv.imshow("MedianBlur Image", thresh2)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()

#     thresh2 = 255-thresh2
#     thresh2 = cv.dilate(thresh2, np.ones(25))
#     thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=3)






#     #*********************Line Segementation**************************************

#     def isLine(list):
#         size= len(list)
#         white,black = countfreq(list)
#         ratio=(white/size,black/size)
#         # print(white,black,ratio)
#         if(ratio[1]>0.97):
#             return 1
#         else:
#             return 0


#     height=thresh2.shape[0]
#     width=thresh2.shape[1]

#     lineornot=[]
#     for i in range(height):
#         lineornot.append(isLine(thresh2[i]))

#     def rectify(list):
#         for i in range(3,len(list)-3,3):
#             if(np.count_nonzero(list[i-3:i+3])<3):
#                 list[i-3:i+3]=[0]*6
#         return list
        
#     lineornot=rectify(lineornot)


#     for i in range(height):
#         if(lineornot[i]==True):
#             thresh2[i]=[255]*width

#     # cv.namedWindow('d', 0)
#     # cv.imshow("d",thresh2)
#     # cv.waitKey(0)


#     thresh2=255-thresh2
#     (components, _) = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     linecounter=1
#     linedistance=0
#     for c in components:
        
#         # skip small word candidates
#         if cv.contourArea(c) < 110000:
#             continue
        
#         currBox = cv.boundingRect(c) # returns (x, y, w, h)
#         (x, y, w, h) = currBox
#         currLine = PageSegmentedImage[y-10:y+h+25, 0:thresh2.shape[1]]
#         if(linedistance==0 and h<280 and h>50):
#             print(f"Change in Line Distance: {h+25}")
#             linedistance=h+25
#         if(h<280 and h>0):
#             # cv.namedWindow('Final Segmented Line', 0)
#             # cv.imshow("Final Segmented Line",currLine)
#             cv.imwrite(f'{path}/Lines/Line{imgcounter}_{linecounter}.png', currLine)
#             # cv.waitKey(0)
#         elif(linedistance>0):
#             roh=round(h/linedistance)
#             print(f"Lines divided into: {h/linedistance} Roh: {roh}")

#             for j,i in enumerate(range(int(h/roh),h,int(h/roh))):
#                 currLinen = currLine[int(h/roh)*j:i+25, 0:w]
#                 # cv.namedWindow('Final Segmented Line Sperated', 0)
#                 # cv.imshow("Final Segmented Line Sperated",currLinen)
#                 # cv.waitKey(0)
#                 try:
#                     cv.imwrite(f'{path}/Lines/Line{imgcounter}_{linecounter}.png', currLinen)
#                 except:
#                     pass
#                 linecounter+=1
#         else:
#             pass
#         cv.destroyAllWindows()
#         linecounter+=1
#     imgcounter+=1


# ****************word segmentation******************      

filename= os.listdir(path+'/Lines')

for name in filename:
    file=path+'/Lines/'+name
    print(file)
    SegmentedLine =cv.imread(file)
 
    gray = cv.cvtColor(SegmentedLine, cv.COLOR_BGR2GRAY)
    ret3, thresh2 = cv.threshold(
    gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # cv.namedWindow('Orignal Segmented Line', 0)
    # cv.imshow("Orignal Segmented Line", thresh2)
    # cv.waitKey(0)

    thresh2 = 255-thresh2
    thresh2 = cv.dilate(thresh2, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
    thresh2 = 255-thresh2


    laplacian = cv.Laplacian(gray, -1, ksize=15)
    minLineLength = 60
    maxLineGap = 60
    lines = cv.HoughLinesP(laplacian,5,np.pi/180,550,minLineLength,maxLineGap)
    # print(f"{type(lines) is type(None)}")
    if(lines is not None):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(thresh2,(x1,y1),(x2,y2),(255,255,255),9)

    # cv.namedWindow('Lines Removed Word', 0)
    # cv.imshow("Lines Removed Word", thresh2)
    # cv.waitKey(0)

    thresh2 = cv.medianBlur(thresh2, 3)

    # cv.namedWindow('MedianBlur Word', 0)
    # cv.imshow("MedianBlur Word", thresh2)
    # cv.waitKey(0)

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

    # cv.namedWindow('Word Seperated', 0)
    # cv.imshow("Word Seperated",thresh2)
    # cv.waitKey(0)
    cv.destroyAllWindows()
    (components, _) = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bBox=[]


    wordcounter=1
    for c in components:
        
        # skip small word candidates
        if cv.contourArea(c) < 5000:
            continue
        
        currBox = cv.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = SegmentedLine[y:y+h, x:x+w]
        cv.namedWindow('final Segemented word', 0)
        cv.imshow("final Segemented word",currImg)
        cv.waitKey(0)

        # cv.imwrite(f"{path}/Words/word{name[4:-4]}_{wordcounter}.png",currImg)
        CharSegmentation.Cycler(currImg,1,cseg,f"{name[4:-4]}_{wordcounter}")
        wordcounter+=1
    cv.destroyAllWindows()