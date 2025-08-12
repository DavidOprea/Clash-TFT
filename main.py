import picture
import vision
import time
import os
import cv2

def main():
    photographer = picture.Photography()
    #screenshot, filename = photographer.takePicture()
    time.sleep(1)

    filename = "screen_2025-08-12_17-47-28.png"

    detect = vision.Vision(filename)
    elixir_img, cardCoors = detect.findTemp("Templates")

    cv2.imshow('Matches Found', elixir_img)
        # --- END OF NEW CODE ---
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(cardCoors)
    print(detect.read_elixir_from_image(elixir_img))

    #photographer.deletePicture(filename)
    
if __name__ == "__main__":
    main()