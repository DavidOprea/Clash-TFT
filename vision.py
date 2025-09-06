import datetime
import os
import cv2
import pytesseract
import numpy as np
import time

class Vision():
    def __init__(self):
        # A lot of variables
        self.img = None
        self.startImg1 = cv2.cvtColor(cv2.imread("start.png"), cv2.COLOR_BGR2RGB)
        self.startImg2 = cv2.cvtColor(cv2.imread("start1.png"), cv2.COLOR_BGR2RGB)
        self.play_again = cv2.cvtColor(cv2.imread("play_again.png"), cv2.COLOR_BGR2RGB)
        self.battle = cv2.cvtColor(cv2.imread("battle.png"), cv2.COLOR_BGR2RGB)
        self.empty_tile = cv2.cvtColor(cv2.imread("tile.png"), cv2.COLOR_BGR2GRAY)
        self.quit = cv2.cvtColor(cv2.imread("quit.png"), cv2.COLOR_BGR2RGB)
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        self.templates = {}
        for filename in os.listdir("Templates"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                template_path = os.path.join("Templates", filename)
                template = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_BGR2RGB)
                if template is not None:
                    # Store the template without extension
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
        max_retries = 5
        retry_delay = 0.5 
        
        # This is in case there is somehow an error in reading the image
        for _ in range(max_retries):
            try:
                img_bgr = cv2.imread(imgPath)
                if img_bgr is not None and not img_bgr.size == 0:
                    self.img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    return
                else:
                    time.sleep(retry_delay)
            except Exception as e:
                time.sleep(retry_delay)

    # Finds the "start" of each round through an icon
    def findStart(self):
        curImg = self.img[500:700, 320:480]
        threshold = 0.64
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.startImg1, method)
        _, max_val1, _, _ = cv2.minMaxLoc(res)
        res = cv2.matchTemplate(curImg, self.startImg2, method)
        _, max_val2, _, _ = cv2.minMaxLoc(res)

        return max(max_val1, max_val2) >= threshold

    # Finds the play again button 
    def findEnd(self):
        curImg = self.img
        threshold = 0.51
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.play_again, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return (max_val >= threshold)
    
    # Finds the quit button
    def findQuit(self):
        curImg = self.img
        threshold = 0.43
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.quit, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val >= threshold

    # Finds the battle button in case the agent somehow goes to the starting menu
    def findBattle(self):
        curImg = self.img
        threshold = 0.45
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(curImg, self.battle, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold
    
    # Determine the rank (1-4) to give reward
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
    
    # Find the health in a round for rewarding purposes
    def findHealth(self):
        curImg = self.img[1020:1060, 745:775]
        curImgGray = cv2.cvtColor(curImg, cv2.COLOR_RGB2GRAY)
        method = cv2.TM_CCOEFF_NORMED
        best_score = 0
        best = 1
        for i in range(len(self.health)):
            res = cv2.matchTemplate(curImgGray, self.health[i], method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best = i+1
                
        return best

    # Find the cards the agent can use
    def findTemps(self):
        #define variables
        curImg = self.img
        threshold = 0.46
        method = cv2.TM_CCOEFF_NORMED

        #returns 1510 675 around 2.1 (scaling window coors to opencv2 coors)
        elixir_img = curImg[1482:1542, 630:720]
        cards_img = curImg[1350:1600, 150:600]
        cardCoors = []

        i = 1

        #check to see which templates are in current game state
        for _, template in self.templates.items():
            _, w, h = template.shape[::-1]
            
            res = cv2.matchTemplate(cards_img, template, method)

            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # Check if the maximum value is above the confidence threshold
            if max_val >= threshold:
                #print(f"Match found for '{name}' with a confidence of {max_val:.2f} at location {max_loc}")
                top_left = max_loc
                x,y = top_left[0], top_left[1]

                cardCoors.append([i, y+(h/2), x+(w/2)])

            i += 1

        """
        # Drawing rectangles for inital testing
        scale_percent = 40

        width = int(curImg.shape[1] * scale_percent / 100)
        height = int(curImg.shape[0] * scale_percent / 100)
        dim = (width, height)

        curImg = cv2.resize(curImg, dim, interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Matches Found', curImg)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        return elixir_img, cardCoors

    # Get the current elixir through template matching
    def read_elixir_from_image(self, elixir_image):
        if elixir_image is None or elixir_image.size == 0:
            print("Error: Input image for digit recognition is empty or invalid.")
            return -1
        
        method = cv2.TM_CCOEFF_NORMED

        cur_max_val = 0
        cur = 0

        elixir_image_gray = cv2.cvtColor(elixir_image, cv2.COLOR_BGR2GRAY)

        # Iterate through all digit templates (0-20)
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
    
    # Tried this, but not necessary
    '''
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
    '''

    def find_occupied_tile_by_color_variation(self, coors):
        size = 30
        worst = [0, None]
        
        print("Analyzing tiles for color variation...")
        
        # Iterate through each of the 20 possible board coordinates
        for i in range(len(coors)):
            x, y = coors[i]
            x = round(x*2.05)
            y = round(y*2.02)
            
            tile = self.img[y - size:y + size, x - size:x + size]
            #cv2.imshow(f"Tile {i}", tile)
            #cv2.waitKey(0)
            
            # Check if the tile is a valid image
            if tile.size == 0:
                print(f"Warning: Tile at index {i} is empty.")
                continue

            b_std = np.std(tile[:,:,0])
            g_std = np.std(tile[:,:,1])
            r_std = np.std(tile[:,:,2])
            
            total_variation = b_std + g_std + r_std
            
            #print(f"Tile {i} at ({x}, {y}): Variation Score = {total_variation:.2f}")

            if total_variation > worst[0]:
                worst[0] = total_variation
                worst[1] = i
        #cv2.destroyAllWindows()
        
        return worst[1]
    
    # For Debugging Purposes and Stuff
    def showImage(self, y1, y2, x1, x2):
        img = self.img[y1:y2, x1:x2]
        
        output_dir = "Health" #I was testing health bars

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created folder: {output_dir}")
        
        # Create a unique filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"health_image_{timestamp}.png"
        file_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(file_path, img)
        
        cv2.imshow('Img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()