import datetime
import os
import cv2
import pytesseract
import numpy as np
import math

class Vision():
    def __init__(self):
        self.img = None
        self.startImg1 = cv2.cvtColor(cv2.imread("start.png"), cv2.COLOR_BGR2RGB)
        self.startImg2 = cv2.cvtColor(cv2.imread("start1.png"), cv2.COLOR_BGR2RGB)
        self.play_again = cv2.cvtColor(cv2.imread("play_again.png"), cv2.COLOR_BGR2RGB)
        self.empty_tile = cv2.cvtColor(cv2.imread("tile.png"), cv2.COLOR_BGR2GRAY)
        self.quit = cv2.cvtColor(cv2.imread("quit.png"), cv2.COLOR_BGR2RGB)
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        self.templates = {}
        for filename in os.listdir("Templates"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                template_path = os.path.join("Templates", filename)
                template = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_BGR2RGB)
                if template is not None:
                    # Store the template with its name (without extension)
                    self.templates[filename[:-4]] = template
        
        self.ranks = []
        for filename in os.listdir("Ranks"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rank_path = os.path.join("Ranks", filename)
                rank = cv2.cvtColor(cv2.imread(rank_path), cv2.COLOR_BGR2RGB)
                if rank is not None:
                    self.ranks.append(rank)
        
        self.health = []
        for filename in os.listdir("Health"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                health_path = os.path.join("Health", filename)
                h = cv2.cvtColor(cv2.imread(health_path), cv2.COLOR_BGR2GRAY)
                if h is not None:
                    self.health.append(h)
    
    def setImg(self, imgPath):
        self.img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

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
        curImg = self.img[1400:1550, 110:460]
        threshold = 0.55
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.play_again, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold
    
    def findQuit(self):
        curImg = self.img[1350:1550, 200:700]
        threshold = 0.43
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.quit, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold

    def findRank(self):
        curImg = self.img[180:420, 320:580]
        method = cv2.TM_CCOEFF_NORMED
        best_score = 0
        best = 1
        for i in range(len(self.ranks)):
            res = cv2.matchTemplate(curImg, self.ranks[i], method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best = i+1
        return best
    
    def findHealth(self):
        curImg = self.img[1020:1060, 745:775]
        curImgGray = cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY)
        cv2.imshow("Image", curImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        method = cv2.TM_CCOEFF_NORMED
        for i in range(len(self.health)):
            res = cv2.matchTemplate(curImgGray, self.health[i], method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            print(f"{i+1}: {max_val}")

    
    def findTemps(self):
        #define variables
        curImg = self.img
        threshold = 0.49
        method = cv2.TM_CCOEFF_NORMED

        #returns 1510 675 around 2.1
        elixir_img = curImg[1482:1542, 630:720]
        cards_img = curImg[1350:1600, 150:600]
        cardCoors = []

        i = 1

        #check to see which templates are in current game state
        for name, template in self.templates.items():
            _, w, h = template.shape[::-1]
            
            # Perform the template matching
            res = cv2.matchTemplate(cards_img, template, method)
            
            # Find the maximum correlation value and its location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Check if the maximum value is above our confidence threshold
            if max_val >= threshold:
                print(f"Match found for '{name}' with a confidence of {max_val:.2f} at location {max_loc}")
                # You can now take action, like drawing a rectangle
                top_left = max_loc
                x,y = top_left[0], top_left[1]
                bottom_right = (x+w, y+h)

                cardCoors.append([i, y+(h/2), x+(w/2)])
                i += 1
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

    def checkBoardForTroop(self, coors):
        size = 15
        curImg = self.img
        worst = [-100, 0]
        for i in range(len(coors)):
            x, y = coors[i][0], coors[i][1]
            tile = curImg[x-size:x+size, y-size:y+size]
            cv2.imshow(f"Tile {i}", tile)
            cv2.waitKey(1)
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            method = cv2.TM_SQDIFF
            res = cv2.matchTemplate(tile_gray, self.empty_tile, method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > worst[0]:
                worst = [max_val, i]
            print(f"Score {i}: {max_val}")
        print(f"Worst {worst[1]}: {worst[0]}")
        cv2.destroyAllWindows()
        return worst[1]

    def find_occupied_tile_by_color_variation(self, coors):
        """
        Finds the coordinates of the troop by calculating the total color
        variation (standard deviation) of each tile. The tile with the highest
        variation is considered the one with the troop.
        
        Returns:
            tuple: A tuple (x, y) of the top-left corner of the troop's tile,
                   or None if no significant variation is found.
        """
        size = 30  # The size of the tile to analyze
        worst = [0, None] # [highest_variation_score, index_of_tile]
        
        print("Analyzing tiles for color variation...")
        
        # Iterate through each of the 20 possible board coordinates
        for i in range(len(coors)):
            x, y = coors[i]
            x = round(x*2.05)
            y = round(y*2.02)
            
            # Extract the tile from the screenshot using the size
            tile = self.img[y - size:y + size, x - size:x + size]
            #cv2.imshow(f"Tile {i}", tile)
            #cv2.waitKey(0)
            
            # Check if the tile is a valid image (not empty)
            if tile.size == 0:
                print(f"Warning: Tile at index {i} is empty.")
                continue

            # Calculate the standard deviation for each color channel (B, G, R)
            b_std = np.std(tile[:,:,0])
            g_std = np.std(tile[:,:,1])
            r_std = np.std(tile[:,:,2])
            
            # Calculate a total color variation score.
            # We use a simple sum as a measure of total variation.
            total_variation = b_std + g_std + r_std
            
            #print(f"Tile {i} at ({x}, {y}): Variation Score = {total_variation:.2f}")

            # Find the tile with the highest color variation
            if total_variation > worst[0]:
                worst[0] = total_variation
                worst[1] = i
        #cv2.destroyAllWindows()
        
        # A simple threshold to avoid false positives on empty tiles with slight variations.
        # You may need to adjust this value based on your game's visuals.
        return worst[1]
    
    # For Debugging Purposes and Stuff
    def showImage(self, y1, y2, x1, x2):
        img = self.img[y1:y2, x1:x2]
        
        output_dir = "Health"
        
        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created folder: {output_dir}")
        
        # Create a unique filename using a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"health_image_{timestamp}.png"
        file_path = os.path.join(output_dir, filename)
        
        # Save the image to the new folder
        cv2.imwrite(file_path, img)
        print(f"Image saved to: {file_path}")
        
        
        cv2.imshow('Img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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