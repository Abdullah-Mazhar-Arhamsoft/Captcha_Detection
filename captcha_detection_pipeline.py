import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from captcha_preprocess import pre_process_image, second_preprocess_captcha
import sqlite3


loaded_model = keras.models.load_model('captcha_recognition')

def save_to_database(predicted_text):
    try:
        # connect to database
        conn = sqlite3.connect('captcha_detection.db')
        cursor = conn.cursor()

        # Create table if doesnt exist
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS captcha_predictions (
                       id INTEGER PRIMARY KEY AUTOINCREMENT, predicted_text TEXT
                       )''')
        
        # Check if predicted text already exists in the database
        cursor.execute('SELECT * FROM captcha_predictions WHERE predicted_text = ?', (predicted_text,))
        existing_record = cursor.fetchone()

        if existing_record is None:
            
            # Insert the text in db
            cursor.execute('INSERT INTO captcha_predictions (predicted_text) VALUES (?)', (predicted_text,))
            conn.commit()

            print("Data Stored Successfully")
        else:
            print("Predicted text already exist in database")

        # Get list of unique predicted texts
        cursor.execute('SELECT DISTINCT predicted_text FROM captcha_predictions')
        unique_records = [row[0] for row in cursor.fetchall()]
        print(unique_records)

        for record in unique_records:
            cursor.execute('SELECT id FROM captcha_predictions WHERE predicted_text = ?', (record,))
            duplicate_records = cursor.fetchall()

            if len(duplicate_records) > 1:
                keep_id = min(duplicate_records)[0]
                cursor.execute('DELETE FROM captcha_predictions WHERE predicted_text = ? AND id != ?', (record, keep_id))

        print("Duplicated records deleted")     
        
        conn.commit()


    except Exception as e:
        print("Error during database operation", str(e))

    finally:
        if conn:
            conn.close()


def preprocess_captcha(image_path, output_image):
    pre_processed_image = pre_process_image(image_path)

    cv2.imwrite(output_image, pre_processed_image)
    return pre_processed_image


def decode_batch_predictions(pred, max_length, num_to_char):
    try:
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    except Exception as e:
        print("Error during decode batch prediction", str(e))
        return None



def captcha_pipeline(image_path, max_length=6):
    try:
        output_image = 'captcha_image.png'

        # Preprocess the captcha image
        captcha_image = preprocess_captcha(image_path, output_image)

        # Second preprocessing step
        new_captcha_image = second_preprocess_captcha(output_image)

        prediction_model_v1 = keras.models.Model(
        loaded_model.get_layer(name="image").input,
        loaded_model.get_layer(name="dense2").output,
        )

        characters = ['3', '6', 'B', '1', 'e', 'b', 'l', 'd', 'K', 'Q', 'E', '9', '5', 'm', 'y', 'R', 'a', 'Y', '2', 'N', 'k', 't', 'W', '7', 'M', '8', 'L', 'A', 'T', 'F', 'H']
        
        char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

        num_to_char = layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

        preds = prediction_model_v1.predict(np.expand_dims(new_captcha_image, axis=0))
        pred_texts = decode_batch_predictions(preds, max_length, num_to_char)

        # Print the predicted text
        if len(pred_texts) > 0:
            print("Predicted Text:", pred_texts[0])
            save_to_database(pred_texts[0])
        else:
            print("No text was predicted.")

    except Exception as e:
        print("Erro during captcha processing", str(e))
        return None

# # Define your model path, image path, and other parameters
image_path = "2LEEEK.png"

# # Run the pipeline
captcha_pipeline(image_path)
