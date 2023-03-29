import os
import requests
import pymysql

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
        return None

def download_image(url, file_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_card_images(conn, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with conn.cursor() as cursor:
        cursor.execute("SELECT id, scryfallId, frameVersion FROM cards")
        results = cursor.fetchall()

        for result in results:
            card_id, scryfall_id, frame_version = result
            if frame_version == "2015":
                image_url = f"https://api.scryfall.com/cards/{scryfall_id}?format=image"

                output_file_path = os.path.join(output_folder, f"{card_id}.jpg")

                try:
                    download_image(image_url, output_file_path)
                    print(f"Downloaded image for card id {card_id} to {output_file_path}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download image for card id {card_id}: {e}")

if __name__ == "__main__":
    conn = connect_to_database()
    if conn:
        output_folder = "card_images"
        download_card_images(conn, output_folder)
        conn.close()