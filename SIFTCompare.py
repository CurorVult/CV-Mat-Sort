import cv2
import numpy as np
import os
import imageProcessing
import altComparison
import pymysql

capture = cv2.VideoCapture(0)
# Get the actual frame dimensions of the webcam
widthImg = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
heightImg = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set Camera Parameters
capture.set(10, 130)
IPKernel = np.ones((5,5))

# Use the calculated frame dimensions
capture.set(3, widthImg)
capture.set(4, heightImg)

def connect_to_database():
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'Mvemjsunp9',
        'database': 'cardscandb'
    }

    try:
        conn = pymysql.connect(**config)
        if conn.open:
            print("Connected to MySQL database")
            return conn
    except pymysql.Error as e:
        print(f"Error connecting to MySQL database: {e}")
        
# Connect to the database
conn = connect_to_database()

while True:
    success, img= capture.read()
    imgCont= img.copy()
    #resize before image preprocessing
    img= cv2.resize(img,(widthImg, heightImg))
    #apply filters
    imgContour= img.copy()
    imgThresh=imageProcessing.imageProcessing(img)
    #find the largest 4 sided "object" in the camera field of view and draw a bounding box around it
    largest = imageProcessing.getContours(imgThresh)
    #Correct for warped/skewed perception
    img_warped = imageProcessing.getWarp(img,largest)

    cv2.imshow("Result",imgThresh)

    # Add this line to break the loop for testing purposes (press 'q' to break)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compare the warped image with images in a folder
folder_path = "path/to/your/folder"
matching_image_name = altComparison.sift_comparison(img_warped, folder_path)

# Get the name from the SQL database
with conn.cursor() as cursor:
    cursor.execute(f"SELECT name FROM cards WHERE card_id = '{matching_image_name}'")
    result = cursor.fetchone()
    if result:
        print("Matching card name:", result[0])
    else:
        print("No match found in the database")

# Close the database connection
conn.close()

# Don't forget to release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()