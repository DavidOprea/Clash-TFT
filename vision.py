import os
import cv2
import pytesseract
import numpy as np

class Vision():
    def __init__(self, imgPath):
        self.img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        self.startImg = cv2.cvtColor(cv2.imread("start.png"), cv2.COLOR_BGR2RGB)
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def findStart(self):
        curImg = self.img.copy()
        threshold = 0.5
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.startImg, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        print(max_val)

        return max_val >= threshold

    
    def findTemp(self, templates):
        #define variables
        curImg = self.img.copy()
        threshold = 0.49
        method = cv2.TM_CCOEFF_NORMED

        #returns
        elixir_img = curImg[1482:1542, 630:720]
        cardCoors = []

        #check to see which templates are in current game state
        for filename in os.listdir(templates):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                template_path = os.path.join(templates, filename)
                template = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_BGR2RGB)
                
                if template is not None:
                    # Get the width and height of the template
                    _, w, h = template.shape[::-1]
                    
                    # Perform the template matching
                    res = cv2.matchTemplate(curImg, template, method)
                    
                    # Find the maximum correlation value and its location
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
                    # Check if the maximum value is above our confidence threshold
                    if max_val >= threshold:
                        print(f"Match found for '{filename}' with a confidence of {max_val:.2f} at location {max_loc}")
                        # You can now take action, like drawing a rectangle
                        top_left = max_loc
                        x,y = top_left[0], top_left[1]
                        bottom_right = (x+w, y+h)
                        cv2.rectangle(curImg, top_left, bottom_right, (0, 255, 0), 2)

                        cardCoors.append([filename[:-4], y+(h/2), x+(w/2)])
        
        # You can now show the image with the matches highlighted
        scale_percent = 40
        # Calculate the new dimensions
        width = int(curImg.shape[1] * scale_percent / 100)
        height = int(curImg.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Resize the image using cv2.resize()
        curImg = cv2.resize(curImg, dim, interpolation=cv2.INTER_AREA)
        

        # You can now show the resized image with the matches highlighted
        """
        cv2.imshow('Matches Found', curImg)
        # --- END OF NEW CODE ---
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        return elixir_img, cardCoors

    def read_elixir_from_image(self, elixir_image):
        if elixir_image is None:
            print("Error: Input image for OCR is None.")
            return -1

        # --- Image Preprocessing for better OCR accuracy ---
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(elixir_image, cv2.COLOR_BGR2GRAY)
        
        # Optional: Apply thresholding to make the numbers clearer
        # The numbers in the elixir bar are often bright, so this can help.
        # Adjust the values (e.g., 200, 255) based on your image.
        _, thresh_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        
        # Optional: Dilate the image to make the text thicker
        kernel = np.ones((2, 2), np.uint8)
        processed_image = cv2.dilate(thresh_image, kernel, iterations=1)
        
        # --- Perform OCR ---
        # Use Pytesseract to extract the number from the processed image
        # The '-c tessedit_char_whitelist=0123456789' part tells Tesseract
        # to only look for digits, which greatly improves accuracy.
        # The '--psm 6' part is a Page Segmentation Mode, which assumes
        # a single uniform block of text.
        config = '-c tessedit_char_whitelist=0123456789 --psm 6'
        number_string = pytesseract.image_to_string(processed_image, config=config).strip()
        
        # --- Convert the result to an integer ---
        try:
            elixir_count = int(number_string)
            return elixir_count
        except ValueError:
            gray_image = cv2.cvtColor(elixir_image, cv2.COLOR_BGR2GRAY)
    
            # Use Otsu's thresholding, which automatically finds the best threshold value
            # This is often more reliable than a fixed threshold.
            _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # --- Perform OCR ---
            # Use Pytesseract with a whitelist for digits.
            # We will stick with psm 6, as it is often correct for this kind of input.
            config = '-c tessedit_char_whitelist=0123456789 --psm 6'
            number_string = pytesseract.image_to_string(processed_image, config=config).strip()
            
            # --- Convert the result to an integer ---
            try:
                elixir_count = int(number_string)
                return elixir_count
            except ValueError:
                print(f"OCR failed to convert '{number_string}' to an integer.")
                return -1