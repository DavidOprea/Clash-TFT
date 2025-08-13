
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
        self.curMax = 1
    
    def decide(self, cards, curElixir):
        for i in range(3):
            if((len(self.on_field) < self.curMax or cards[i][0] in self.on_fields) and 
               self.costs[cards[i][0]] <= curElixir):
                return self.coors[i], self.costs[cards[i][0]]
        return -1, 0