import pydirectinput
import pygetwindow as gw
import time

class Mouse():
    def __init__(self):
        self.window = gw.getWindowsWithTitle("LDPlayer")[0]

    def drag_card(self, start_x, start_y, end_x, end_y, steps=5, duration=0.3):
        abs_x = self.window.left
        abs_y = self.window.top

        start_x += abs_x
        end_x += abs_x
        start_y += abs_y
        end_y += abs_y

        # Move to the starting position
        pydirectinput.moveTo(start_x, start_y)
        time.sleep(0.1)
        
        # Press and hold the left mouse button
        pydirectinput.mouseDown(button='left')
        time.sleep(0.1)
        
        # Calculate the incremental steps
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps
        
        step_duration = duration / steps
        
        # Move the mouse in small steps
        for i in range(1, steps + 1):
            pydirectinput.moveTo(int(start_x + i * dx), int(start_y + i * dy))
            time.sleep(step_duration)
            
        # Release the mouse button
        pydirectinput.mouseUp(button='left')
    
    def left_click(self, x, y):
        pydirectinput.click(x, y)