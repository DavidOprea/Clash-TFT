import picture
import vision
import mouse_control
import strategizer
import time
import os
import cv2

def main():
    #Set up everything
    photographer = picture.Photography()
    mouse = mouse_control.Mouse()
    decider = strategizer.Decider()
    detect = vision.Vision()
    curElixir = 4
    sell = True
    start = False

    #mouse.drag_card(120, 701, 235, 500)
    #screenshot, filename = photographer.takePicture()

    for _ in range(1000):
        while True:
            #120, 701
            
            filename = photographer.takePicture()
            detect.setImg(filename)

            print("OK")
            ans = input()

            #1020, 1060, 745, 775
            detect.findHealth()
        
            photographer.deletePicture(filename)

            '''
            if detect.findStart():
                start = True
                if sell:
                    loc = detect.find_occupied_tile_by_color_variation(decider.board)
                    decider.sellTroop(loc)
                sell = False
            elif detect.findEnd():
                mouse.left_click(150, 700)
                sell = True
                decider.reset()
            elif detect.findQuit():
                sell = True
                decider.reset()
                mouse.left_click(150, 700)
            else:
                if start:
                    decider.upMax()
                start = False
                photographer.deletePicture(filename)
                continue

            photographer.deletePicture(filename)

            filename = photographer.takePicture()
            detect.setImg(filename)
            elixir_img, cardCoors = detect.findTemps()

            path = os.path.join("Elixir Templates", filename)

            cv2.imwrite(path, elixir_img)

            cardCoors = sorted(cardCoors, key=lambda x: x[2])

            if(len(cardCoors) == 0):
                continue

            print(cardCoors)
            
            if detect.read_elixir_from_image(elixir_img) != -1:
                curElixir = detect.read_elixir_from_image(elixir_img)
            
            print("Cur Elixir: " + str(curElixir))
            
            decision, sub = decider.decide(cardCoors, curElixir)

            if(decision == -1):
                break

            curElixir -= sub
                
            mouse.drag_card(decision[0], decision[1], decision[0], decision[1]-200)


            photographer.deletePicture(filename)
            '''
        
        time.sleep(15)
        decider.upMax()
    
    
if __name__ == "__main__":
    main()