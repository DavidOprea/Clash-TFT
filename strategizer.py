import mouse_control
import vision

class Decider():
    def __init__(self):
        self.coors = [[120, 701], [180,701], [240,701]]
        self.costs = {"knight" : 2, "archer" : 2, "goblin" : 2, 
                      "spear goblin" : 2, "bomber" : 2,
                      "barbarian" : 2, "valkyrie" : 3, 
                      "pekka" : 3, "prince" : 3, "giant skeleton" : 3, 
                      "dart goblin" : 3, "executioner" : 3, "princess" : 4,
                      "mega knight" : 4, "royal ghost" : 4, "bandit" : 4, 
                      "goblin machine" : 4, "skeleton king" : 5,
                      "golden knight" : 5, "archer queen" : 5}
        self.on_field = []
        self.board = [[135, 440], [180, 440], [225, 440], [270, 440], [315, 440], 
                      [155, 470], [200, 470], [245, 470], [290, 470], [335, 470], 
                      [135, 500], [180, 500], [225, 500], [270, 500], [315, 500],
                      [175, 530], [200, 530], [245, 530], [290, 530], [335, 530]]
        self.curMax = 2
        self.mouse = mouse_control.Mouse()
    
    def decide(self, cards, curElixir):
        for i in range(len(cards)):
            if((len(self.on_field) < self.curMax or cards[i][0] in self.on_field) and 
               self.costs[cards[i][0]] <= curElixir):
                if(cards[i][0] not in self.on_field):
                    self.on_field.append(cards[i][0])
                return self.coors[i], self.costs[cards[i][0]]
        return -1, 0
    
    def upMax(self):
        self.curMax += 1
    
    def sellTroop(self, loc):
        spots = [[325, 200], [325, 250], [325, 300], [325, 350], [325, 400], [325, 450], [325, 500]]
        self.mouse.left_click(self.board[loc][0], self.board[loc][1])
        for spot in spots:
            self.mouse.left_click(spot[0], spot[1])
    
    def reset(self):
        self.curMax = 2
        self.on_field = []