import cv2
import numpy as np
# Set Camera Parameters
#Code adapted from computervision.zone's 3 Hour Computer Vision workshop
widthImg =640
heightImg =480
capture= cv2.VideoCapture(1)
capture.set(3,widthImg)
capture.set(4, heightImg)
capture.set(10,130)
IPKernel=np.ones((5,5))


def imageProcessing(img):
    #while we will need a colour image later on, for now we convert to greyscale to reduce extrenious data when identifying the card itself
    imgGrey = cv2.cvtColor(img,cv2,cv2.COLOR_BGR2GRAY)
    #To further reduce noise, we use gaussian blur to reduce hard edges, we use a 5 by 5 kernel
    imgBlur = cv2.GaussianBlur(imgGrey, (5,5),1)
    #Next we pick out the edges using Canny Edge detector 
    imgCanny = cv2.Canny(imgBlur,200,200)
    #We thicken the edges using the dilation function, we do not need to trace fine detailed shapes. We want to find the rectangular bounds of the card
    #We make 2 passes of dilation to create very thick lines, then cut them back with one pass of erosion. This method was reccomended by Murtaza's Workshop
    imgDial=cv2.dialate(imgCanny,IPKernel,iterations=2)
    imgErode= cv2.erode(imgDial,IPKernel, iterations=1)
    return imgErode


def getContours(img):
    largest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>5000:
            #cv2.drawContours(imgContour,cnt,-1,(255,0,0), 3)
            peri = cv2.arcLength(cnt,True)
            aprx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #Here we find the contour with the largest area by looping over every bound area with 4 sides in the frame
            if area > maxArea and len(aprx)==4:
                largest=aprx
                maxArea=area
    #cv2.drawContours(imgContour, largest,-1,(255,0,0), 20)        
    return largest
def reorder(points):
    #This function reorders the array of points to ensure that proper bounding boxes are created
    points= points.reshape((4,3))
    pass




def getWarp(img,largest):
#define two sets of points 
    pts1=np.float32(largest)
    pts2= np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOut= cv2.warpPerspective(img, matrix, (widthImg,heightImg))

    return imgOut




while True:
    success, img= capture.read()
    imgCont= img.copy()
    #resize before image preprocessing
    img= cv2.resize(img,(widthImg, heightImg))
    #apply filters
    imgContour= img.copy()
    imgThresh=imageProcessing(img)
    largest = getContours(imgThresh)
    getWarp(img,largest)

    cv2.imshow("Result",imgThresh)
    #Quit feed, will remove in final result
    if cv2.waitKey(1) & OxFF==ord('q'):
        break

