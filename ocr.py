# ocr.py
import cv2
import numpy as np
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'.venv\Lib\tesseract.exe'

def refine_image(image_path, output_path):

    img = cv2.imread(image_path)
    if img is None:
        return None
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a convolution-based sharpening filter to enhance text details
    sharpening_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpening_kernel)

    # Binarize the sharpened image using Otsu's thresholding
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use median blur to reduce noise
    refined = cv2.medianBlur(thresh, 3)

    cv2.imwrite(output_path, refined)
    return output_path


def extract_text(image_path):
    """
    Extract text from the refined image using Tesseract OCR.
    """
    image = Image.open(image_path)
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config, lang='Devanagari')
    return text
