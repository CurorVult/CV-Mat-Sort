import requests
import os

url_base = "https://www.mtgpics.com/pics/big/one/"
output_folder = "C:/Users/Sean/Documents/GitHub/CV-Mat-Sort/imgs"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through the range of image numbers you want to download
for i in range(1, 480):
    img_number = f"{i:03d}"
    img_url = f"{url_base}{img_number}.jpg"
    
    # Download the image using requests
    response = requests.get(img_url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the image to the output folder
        img_path = os.path.join(output_folder, f"{img_number}.jpg")
        with open(img_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"Downloaded image: {img_number}.jpg")
    else:
        print(f"Failed to download image: {img_number}.jpg")