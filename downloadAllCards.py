import requests
import os
import json
import time

output_folder = "C:/Users/Sean/Documents/GitHub/CV-Mat-Sort/imgs"
api_url_base = "https://api.scryfall.com/cards"
output_folder = os.path.join(output_folder, "normal")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def download_card_image(card_data, output_folder):
    card_name = card_data["name"]
    image_url = card_data["image_uris"]["normal"]
    img_file_name = f"{card_name}.jpg".replace("/", "-")  # Replace '/' with '-' to avoid creating subdirectories
    img_path = os.path.join(output_folder, img_file_name)

    response = requests.get(image_url)

    if response.status_code == 200:
        with open(img_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"Downloaded image: {card_name}")
    else:
        print(f"Failed to download image: {card_name}")

def fetch_cards(url, output_folder):
    response = requests.get(url)
    data = json.loads(response.text)

    if "data" in data:
        for card in data["data"]:
            if "image_uris" in card:
                download_card_image(card, output_folder)

    if data.get("has_more"):
        time.sleep(0.1)  # Introduce a delay to avoid rate limits
        next_page_url = data["next_page"]
        fetch_cards(next_page_url, output_folder)

fetch_cards(api_url_base, output_folder)
