import cv2
import numpy as np
import os






def sift_comparison(img_query, folder_path):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for the query image
    kp_query, des_query = sift.detectAndCompute(img_query, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    best_match = None
    best_match_score = float('inf')

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find keypoints and descriptors for the train image
            kp_train, des_train = sift.detectAndCompute(img_train, None)

            # Match keypoints
            matches = bf.match(des_query, des_train)
            matches.sort(key=lambda x: x.distance)

            # Calculate the total distance of the top matches
            top_matches = matches[:min(10, len(matches))]
            match_score = sum([match.distance for match in top_matches])

            # Update the best match
            if match_score < best_match_score:
                best_match = file
                best_match_score = match_score

    return os.path.splitext(best_match)[0]


while True:
    success, img= capture.read()
    imgCont= img.copy()
    #resize before image preprocessing
    img= cv2.resize(img,(widthImg, heightImg))
    #apply filters
    imgContour= img.copy()
    imgThresh=imageProcessing(img)
    #find the largest 4 sided "object" in the camera field of view and draw a bounding box around it
    largest = getContours(imgThresh)
    #Correct for warped/skewed perception
    img_warped = getWarp(img,largest)

    cv2.imshow("Result",imgThresh)

    # Add this line to break the loop for testing purposes (press 'q' to break)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compare the warped image with images in a folder
folder_path = "path/to/your/folder"
matching_image_name = sift_comparison(img_warped, folder_path)
print("Matching image name:", matching_image_name)

# Don't forget to release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()