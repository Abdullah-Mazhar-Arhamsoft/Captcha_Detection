import numpy as np
import cv2
import tensorflow as tf

def pre_process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Image not found or cannot be read.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to binarize the image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)

        # Find and remove horizontal and vertical lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

        removed_horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        removed_vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        inverse_binary = cv2.bitwise_not(binary)

        no_lines = cv2.bitwise_and(inverse_binary, inverse_binary, mask=cv2.bitwise_not(removed_horizontal_lines | removed_vertical_lines))

        no_lines = cv2.bitwise_not(no_lines)
        
        ret, thresh = cv2.threshold(no_lines, 127, 255, cv2.THRESH_OTSU)
        dkernel = np.ones((2, 2), np.uint8)
        threshed_dilated = cv2.dilate(thresh, dkernel, iterations=3)
        ekernel = np.ones((3,3), np.uint8)
        threshed_eroded = cv2.erode(threshed_dilated, ekernel, iterations=3)

        thresh = cv2.threshold(threshed_eroded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            if cv2.contourArea(c) < 110:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
        
        result = 255 - thresh

        return result

    except Exception as e:
        print("Error during first image preprocessing", str(e))
        return None


def second_preprocess_captcha(img_path):
    try:
        img_width = 200
        img_height = 50

        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])

        return img
    
    except Exception as e:
        print("Error during second image preprocessing", str(e))
        return None