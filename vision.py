import os
import cv2
import pytesseract
import numpy as np

class Vision():
    def __init__(self, imgPath):
        self.img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        self.startImg1 = cv2.cvtColor(cv2.imread("start.png"), cv2.COLOR_BGR2RGB)
        self.startImg2 = cv2.cvtColor(cv2.imread("start1.png"), cv2.COLOR_BGR2RGB)
        self.play_again = cv2.cvtColor(cv2.imread("play_again.png"), cv2.COLOR_BGR2RGB)
        self.quit = cv2.cvtColor(cv2.imread("quit.png"), cv2.COLOR_BGR2RGB)
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def findStart(self):
        curImg = self.img
        threshold = 0.55
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.startImg1, method)
        _, max_val1, _, _ = cv2.minMaxLoc(res)
        res = cv2.matchTemplate(curImg, self.startImg2, method)
        _, max_val2, _, _ = cv2.minMaxLoc(res)

        return max(max_val1, max_val2) >= threshold

    def findEnd(self):
        curImg = self.img
        threshold = 0.55
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.play_again, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold
    
    def findQuit(self):
        curImg = self.img
        threshold = 0.43
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.quit, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold

    
    def findTemp(self, templates):
        #define variables
        curImg = self.img
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
        """
        # You can now show the image with the matches highlighted
        scale_percent = 40
        # Calculate the new dimensions
        width = int(curImg.shape[1] * scale_percent / 100)
        height = int(curImg.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Resize the image using cv2.resize()
        curImg = cv2.resize(curImg, dim, interpolation=cv2.INTER_AREA)
        

        # You can now show the resized image with the matches highlighted
        
        cv2.imshow('Matches Found', curImg)
        # --- END OF NEW CODE ---
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        return elixir_img, cardCoors

    
    def read_elixir_from_image(self, elixir_image):
        """
        Reads the number from a cropped image by matching individual digit templates.
        """
        if elixir_image is None or elixir_image.size == 0:
            print("Error: Input image for digit recognition is empty or invalid.")
            return -1
        
        method = cv2.TM_CCOEFF_NORMED

        cur_max_val = 0
        cur = 0

        elixir_image_gray = cv2.cvtColor(elixir_image, cv2.COLOR_BGR2GRAY)

        # Iterate through all digit templates
        for i in range(21):
            template_path = os.path.join("Elixir Templates", f"{i}.png")
            if not os.path.exists(template_path):
                print(f"Template for digit {i} not found. Skipping.")
                continue

            template = cv2.imread(template_path, 0)
            if template is None:
                continue

            res = cv2.matchTemplate(elixir_image_gray, template, method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > cur_max_val:
                cur_max_val = max_val
                cur = i
        
        return cur
            

    '''
    def read_elixir_from_image(self, elixir_image):
        """
        Reads the number from a cropped image using improved OCR techniques.
        This version is more robust by using a reliable thresholding method
        and avoiding unnecessary processing that can harm accuracy.

        Args:
            elixir_image (numpy.ndarray): The cropped image containing the elixir number.

        Returns:
            int: The recognized elixir count, or -1 if recognition fails.
        """
        if elixir_image is None or elixir_image.size == 0:
            print("Error: Input image for OCR is empty or invalid.")
            return -1

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(elixir_image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's thresholding for optimal contrast.
        # THRESH_BINARY_INV is used because it's a common pattern for white-on-dark text.
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Resize the image to help Tesseract. This can be more effective than dilation.
        resized_image = cv2.resize(processed_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # --- Perform OCR ---
        # Use a whitelist to only recognize digits and assume a single block of text.
        config = '-c tessedit_char_whitelist=0123456789 --psm 6'
        number_string = pytesseract.image_to_string(resized_image, config=config).strip()
        
        # --- Convert the result to an integer ---
        try:
            if number_string:
                elixir_count = int(number_string)
                return elixir_count
            else:
                return -1 # Return -1 if no number is recognized
        except ValueError:
            print(f"OCR failed to convert '{number_string}' to an integer.")
            return -1
    '''