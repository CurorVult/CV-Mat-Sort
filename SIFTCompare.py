import cv2
import numpy as np
import os
import imageProcessing
import altComparison
import pymysql
import tkinter as tk
from tkinter import ttk

capture = cv2.VideoCapture(1)
# Get the actual frame dimensions of the webcam
widthImg = 640
heightImg = 480

# Set Camera Parameters
capture.set(10, 130)
#Define Kernal for Image Processing Passes
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

#Selection UI for Selecting Feature Matching Method
def on_combobox_change(event):
    global feature_matching_method
    feature_matching_method = combobox.get()

root = tk.Tk()
root.title("Feature Matching Method Selection")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label = ttk.Label(frame, text="Select feature matching method:")
label.grid(row=0, column=0, sticky=tk.W)

feature_matching_methods = ['SIFT', 'SURF', 'ORB', 'AKAZE']
feature_matching_method = tk.StringVar()
combobox = ttk.Combobox(frame, textvariable=feature_matching_method)
combobox['values'] = feature_matching_methods
combobox.grid(row=0, column=1, sticky=tk.W)
combobox.current(0)  # Set the initial value to the first item
combobox.bind("<<ComboboxSelected>>", on_combobox_change)

button = ttk.Button(frame, text="OK", command=root.destroy)
button.grid(row=1, column=0, columnspan=2)

root.mainloop()

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
    img_warped = imageProcessing.getWarp(img, largest, widthImg, heightImg)

    img_with_contour = img.copy()
    if largest is not None and len(largest) > 0:
        cv2.drawContours(img_with_contour, [largest], -1, (0, 255, 0), 2)
    cv2.imshow("Result", img_with_contour)


    # Add this line to break the loop for testing purposes (press 'q' to break)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compare the warped image with images in a folder
folder_path = '\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images'
matching_image_name = altComparison.comparison_by_method(img_warped, folder_path, feature_matching_method)

# Get the name from the SQL database
with conn.cursor() as cursor:
    cursor.execute(f"SELECT name FROM cards WHERE id = '{matching_image_name}'")
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