import cv2
import os

def perform_sift_on_directory(image, directory_path):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors for the input image
    kp1, des1 = sift.detectAndCompute(image, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)

        # Read the image from the directory using cv2.imread
        img2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is read correctly
        if img2 is None:
            print(f"Failed to read image: {file_path}")
            continue

        # Find keypoints and descriptors for the image in the directory
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        # Draw matches and display the result
        img_matches = cv2.drawMatchesKnn(image, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f'Matching result for {file}', img_matches)
        cv2.waitKey(0)

    cv2.destroyAllWindows()