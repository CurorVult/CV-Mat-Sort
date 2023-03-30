import cv2
import os
import numpy as np
import pymysql
from PIL import Image
import imagehash

# Connect to MySQL database
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

# Store AKAZE features and hash in the database
def store_akaze_features_and_hash(conn, image_id, features, akaze_hash):
    cursor = conn.cursor()
    try:
        # Convert AKAZE features to a string representation
        features_str = ",".join(map(str, features.flatten()))
        
        # Update the row in the cards table
        query = f"UPDATE cards SET akaze = '{features_str}', akaze_hash = '{akaze_hash}' WHERE id = '{image_id}'"
        cursor.execute(query)
        conn.commit()
        print(f"Storing hash for image {image_id}: {akaze_hash}")
    except pymysql.Error as e:
        print(f"The error '{e}' occurred")

def extract_akaze_features_and_hash(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    akaze = cv2.AKAZE_create()
    _, descriptors = akaze.detectAndCompute(image, None)
    img_hash = calculate_image_hash(image_path)

    return descriptors, img_hash

# Calculate image hash
def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        img_hash = imagehash.phash(img)
    return str(img_hash)

# Main function
def main():
    folder_path = "\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images"

    # Process images
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_id, _ = os.path.splitext(filename)
            image_path = os.path.join(folder_path, filename)

            print(f"Processing image {filename}")
            akaze_features, img_hash = extract_akaze_features_and_hash(image_path)

            if akaze_features is not None:
                store_akaze_features_and_hash(conn, image_id, akaze_features, img_hash)
            else:
                print(f"Could not extract AKAZE features for image {filename}")

    conn.close()

if __name__ == "__main__":
    main()
