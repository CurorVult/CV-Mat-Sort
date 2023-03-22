import pymysql
import requests
import os

def download_card_images(set_name, host='localhost', port=3306, user='root', password='Mvemjsunp9', database='cardscandb'):
    # Connect to the local SQL database
    connection = pymysql.connect(host=host, port=port, user=user, password=password, database=database)
    print("Connecting to Database")
    try:
        with connection.cursor() as cursor:
            # Obtain the mtgo code from the 'sets' table
            sql_query = "SELECT mtgoCode FROM `sets` WHERE name = %s;"
            cursor.execute(sql_query, (set_name,))
            result = cursor.fetchone()
            if not result:
                print(f"No matching set found for set name: {set_name}")
                return
            
            mtgo_code = result[0]
            
            # Find cards with a matching mtgo code
            sql_query = "SELECT id, scryfallId FROM `cards` WHERE setCode = %s;"
            cursor.execute(sql_query, (mtgo_code,))
            card_ids_and_scryfall_ids = cursor.fetchall()

        # Create a directory for the images
        safe_set_name = set_name.replace(':', '_')  # Replace colon with underscore
        image_directory = os.path.join(os.getcwd(), f"{safe_set_name}_images")
        os.makedirs(image_directory, exist_ok=True)

        # Download images using Scryfall API
        for card_id, scryfall_id in card_ids_and_scryfall_ids:
            url = f"https://api.scryfall.com/cards/{scryfall_id}?format=image&version=normal"
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(os.path.join(image_directory, f"{card_id}.jpg"), 'wb') as img_file:
                    img_file.write(response.content)
            else:
                print(f"Error downloading image for Scryfall ID: {scryfall_id}")

    finally:
        connection.close()

download_card_images('Dissension')