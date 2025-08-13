import picture
import vision
import mouse_control
import strategizer
import time
import os
import cv2

def main():
    photographer = picture.Photography()
    mouse = mouse_control.Mouse()
    decider = strategizer.Decider()
    curElixir = 0

    #mouse.drag_card(120, 701, 235, 500)
    #screenshot, filename = photographer.takePicture()

    for _ in range(10):
        while True:
            filename = photographer.takePicture()
            
            detect = vision.Vision(filename)
            if detect.findStart():
                break
            else:
                photographer.deletePicture(filename)
        
        while True:
            elixir_img, cardCoors = detect.findTemp("Templates")

            cardCoors = sorted(cardCoors, key=lambda x: x[2])

            print(cardCoors)
            
            if detect.read_elixir_from_image(elixir_img) != -1:
                curElixir = detect.read_elixir_from_image(elixir_img)
            else:
                curElixir += 4
            
            decision, sub = decider.decide(cardCoors, curElixir)

            if(decision == -1):
                break

            curElixir -= sub
                
            mouse.drag_card(decision[0], decision[1], decision[0], decision[1]-200)

            photographer.deletePicture(filename)
        
        time.sleep(10)
    
    
if __name__ == "__main__":
    main()