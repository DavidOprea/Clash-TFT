
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
        self.curMax = 4
    
    def decide(self, cards, curElixir):
        print(self.on_field)
        for i in range(len(cards)):
            if((len(self.on_field) < self.curMax or cards[i][0] in self.on_field) and 
               self.costs[cards[i][0]] <= curElixir):
                if(cards[i][0] not in self.on_field):
                    self.on_field.append(cards[i][0])
                return self.coors[i], self.costs[cards[i][0]]
        return -1, 0
    
    def upMax(self):
        self.curMax += 1