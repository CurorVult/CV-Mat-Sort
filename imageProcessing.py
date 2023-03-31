import cv2
import numpy as np
import os

IPKernel = np.ones((5,5))


def imageProcessing(img):
    #Convert Image to Greyscale
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgGrey = clahe.apply(imgGrey)
    #Gaussian Blur filter to reduce noise
    imgBlur = cv2.GaussianBlur(imgGrey, (5,5),1)
    #Find Edges with Canny Edge detection
    imgCanny = cv2.Canny(imgBlur,200,200)
    #Thicken the edges using the dilation function
    imgDial=cv2.dilate(imgCanny,IPKernel,iterations=2)
    #Thin edges with one pass of erosion. This method was reccomended by Murtaza's Workshop
    imgErode= cv2.erode(imgDial,IPKernel, iterations=1)
    return imgErode


def getContours(img):
    largest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>5000:
            #Determine the shape of each contour
            peri = cv2.arcLength(cnt,True)
            aprx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #Find the contour with the largest area by looping over every bound area with 4 sides in the frame
            if area > maxArea and len(aprx)==4:
                largest=aprx
                maxArea=area
    #cv2.drawContours(imgContour, largest,-1,(255,0,0), 20)        
    return largest


def reorder(points):
    #This function reorders the array of points to ensure that proper bounding boxes are created
    points= points.reshape((4,2))
    newPoints=np.zeros((4,1,2),np.int32)
    #Sum points along first axis
    add = points.sum(1)
    #Assign smallest point as the top left corner and the largest as bottom right.
    newPoints[0]=points[np.argmin(add)]
    newPoints[3]=points[np.argmax(add)]
    #Assign the difference between the two middle points as the bottom left and top right
    diff=np.diff(points,axis=1)
    newPoints[1]=points[np.argmin(diff)]
    newPoints[2]=points[np.argmax(diff)]
    return newPoints

    




def getWarp(img, largest, widthImg, heightImg):
    if largest.size == 0:
        return img

    largest = reorder(largest)

    #Assign first set of points
    pts1 = np.float32(largest)
    #Assign second set of points based on passed parameters
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

    #Obtain perspective transformation matrix and use to warp perspective
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    #Correct image but cutting out extranious black areas
    imgCrop = imgOut[10:imgOut.shape[0] - 10, 10:imgOut.shape[1] - 10]
    imgCrop = cv2.resize(imgCrop, (488, 680))

    return imgCrop
