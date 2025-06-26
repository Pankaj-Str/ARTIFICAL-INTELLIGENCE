
### Example: Read Text from an Image Using Python

```python
# Import necessary libraries
import cv2
import pytesseract

# Ensure Tesseract-OCR is installed on your system and its path is set correctly
# Example for Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'example_image.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

# Convert the image to grayscale (optional, improves OCR accuracy)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray_image)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)
```

### Explanation:
1. **Image Loading**: The image is loaded into memory using OpenCV.
2. **Grayscale Conversion**: Converting the image to grayscale can enhance text recognition accuracy.
3. **Text Extraction**: The `image_to_string` function of `pytesseract` extracts text from the image.

Make sure to install the required libraries:
```bash
pip install pytesseract opencv-python
```

Additionally, install Tesseract-OCR on your system. Instructions can be found [here](https://github.com/tesseract-ocr/tesseract).

