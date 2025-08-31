'''
These are the card/team encodings to their corresponding number
1 : archer queen
2 : archer
3 : bandit
4 : barbarian
5 : bomber
6 : dart goblin
7 : executioner
8 : giant skeleton
9 : goblin machine
10 : goblin
11 : golden knight
12 : knight
13 : mega knight
14 : pekka
15 : prince
16 : princess
17 : royal ghost
18 : skeleton king
19 : spear goblin
20 : valkyrie

0: Ranger
1: Ace
2: Clan
3: Goblin
4. Noble
5. Undead
6. Assassin
7. Avenger
8. Brawler
9. Juggernaut
10.Thrower
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from vision import Vision
from picture import Photography
from strategizer import Decider
import mouse_control

class ClashMergeEnv(gym.Env):
    """
    A custom environment for the Clash Merge Tactics game that follows the gym interface.
    This class connects the game's state to a reinforcement learning algorithm.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self):
        super(ClashMergeEnv, self).__init__()

        # Initialize the Photographer
        self.photographer = Photography()
        self.filename = None
        
        # Initialize the Vision class from your existing code.
        self.vision = Vision()

        # Initialize the mouse control
        self.mouse = mouse_control.Mouse()

        # Initialize the strategy stuff
        self.decider = Decider()

        # Initialize board state
        self.board_state = np.zeros(18, dtype=np.int64) 
        self.coors = [[120, 701], [180,701], [240,701]]
        self.board = [[135, 445], [180, 445], [225, 445], [270, 445], [315, 445], 
                      [155, 470], [200, 470], [245, 470], [290, 470],
                      [135, 500], [180, 500], [225, 500], [270, 500], [315, 500],
                      [175, 530], [200, 530], [245, 530], [290, 530]]
        self.card_traits = {1: [2, 7], 2: [0, 2], 3: [1, 7], 4: [2, 8], 5: [5, 10], 6: [3, 0], 
                           7: [1, 10], 8: [5, 8], 9: [3, 9], 10: [3, 6], 11: [4, 6], 12: [4, 9], 
                           13: [1, 8], 14: [1, 9], 15: [4, 8], 16: [0, 4], 17: [5, 6], 
                           18: [5, 9], 19: [3, 10], 20: [2, 7]}
        self.costs = {12 : 2, 2 : 2, 10 : 2, 
                      19 : 2, 5 : 2,
                      4 : 2, 20 : 3, 
                      14 : 3, 15 : 3, 8 : 3, 
                      6 : 3, 7 : 3, 16 : 4,
                      13 : 4, 17 : 4, 3 : 4, 
                      9 : 4, 18 : 5,
                      11 : 5, 1 : 5}
        self.combo_counts = {i: 0 for i in range(11)}
        self.threes = [0, 6, 7, 10] # the unique teams which only work with threes, others are twos and fours. 
        self.sell = True
        self.start = True
        self.cardLimit = 2
        self.curCards = 0
        
        # Define the state space (what the agent sees).
        # We'll use a simple multi-dimensional array for now.
        # This will contain elixir count, card state, and a board representation.
        # The numbers here are placeholders; you'll need to determine the correct dimensions.
        self.observation_space = spaces.Dict({
            "elixir": spaces.Box(0, 20, shape=(1,), dtype=np.float32),
            "hand_cards": spaces.MultiDiscrete([21, 21, 21]),  # Assuming 20 unique cards
            "board_state": spaces.Box(low=0, high=100, shape=(18,), dtype=np.int64) # 5, 4, 5, 4
        })

        # --- UPDATED: Action Space with "Do Nothing" Action ---
        # We now have a larger discrete space to include the "wait" action.
        # The total number of play card actions (5 * 4 * 10) plus one for "do nothing".
        self.total_card_actions = 18 * 3
        self.action_space = spaces.Discrete(self.total_card_actions + 1)

        self.current_state = None
        self.current_health = 9

        # --- NEW: Track the number of actions taken this turn ---
        self.actions_this_turn = 0
        # --- NEW: Track the reward accumulated within the current turn ---
        self.turn_reward = 0

    def _get_obs(self):
        """
        Gathers and processes the current game state from the Vision class.
        This is where you would call your vision methods.
        """
        # Call your vision methods to get the game data.
        self.filename = self.photographer.takePicture()
        self.vision.setImg(self.filename)

        elixir_img, cardCoors = self.vision.findTemps()

        if not cardCoors or len(cardCoors) < 3:
            return {
                "elixir": 0,
                "hand_cards": np.array([0, 0, 0]),
                "board_state": np.zeros(self.observation_space["board_state"].shape, dtype=np.uint64)
            }

        cardCoors = sorted(cardCoors, key=lambda x: x[2])
        elixir = self.vision.read_elixir_from_image(elixir_img)
        
        # Return a dictionary matching the observation_space definition.
        return {
            "elixir": np.array([float(elixir)]),
            "hand_cards": np.array([cardCoors[0][0], cardCoors[1][0], cardCoors[2][0]]), # Placeholder for card recognition
            "board_state": self.board_state
        }
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        This method is called at the beginning of each new game.
        """
        super().reset(seed=seed)
        
        # Simulate a game reset. Here you would trigger the "start game" button.
        # For a real implementation, you would use your Vision class to click the start button.
        # For now, we'll just set up an initial state.

        self.current_health = 9
        self.start = True
        self.sell = True
        self.cardLimit = 2
        self.curCards = 0
        observation = self._get_obs()
        info = {}
        self.board_state = np.zeros(18, dtype=np.int8) 
        return observation, info

    def step(self, action):
        """
        Performs an action in the environment and returns the new state, reward, and other info.
        """

        terminated = False
        reward = -0.02

        #print(self.curCards)
        #print(self.cardLimit)

        if self.vision.findQuit():
            # If the game is over, the agent can't do anything.
            reward = 0
            self.mouse.left_click(150, 700) # Click the quit
            if not self.sell:
                reward = self._get_reward_for_round(True) # Find the rank
            self.mouse.left_click(150, 700) # Click the play again button
            if self.filename:
                self.photographer.deletePicture(self.filename)
            return self.current_state, reward, True, False, {}
        elif self.vision.findEnd():
            reward = 0
            if not self.sell:
                reward = self._get_reward_for_round(True) # Find the rank
            self.mouse.left_click(150, 700) # Click the play again button
            if self.filename:
                self.photographer.deletePicture(self.filename)
            return self.current_state, reward, True, False, {}
        elif self.vision.findBattle():
            reward = 0
            for _ in range(61):
                self.mouse.left_click(250, 600) # Click the battle button
            if self.filename:
                self.photographer.deletePicture(self.filename)
            return self.current_state, reward, True, False, {}
        
        findStart = self.vision.findStart()
        # --- UPDATED: Handle the "Do Nothing" action ---
        if action == self.total_card_actions or (not findStart):
            print("Agent chose to end turn.")
            # This is where the game would advance to the next round.
            # We calculate and return the total reward for the round.
            # reward = self._get_reward_for_round() - I'LL DO THIS IN MAIN PROBABLY
            reward = 0

            if not findStart:
                self.start = True
            
            # Reset counters for the next turn.
            self.actions_this_turn = 0
        elif self.current_state and self.current_state["hand_cards"][0] != 0:
            # Here's where you would translate the action index into a game command.
            # For example, if action = 10, you might play the second card at the third location.
            # You would use your mouse_drag_script to perform the action.
            reward = 1
            if self.sell:
                loc = self.vision.find_occupied_tile_by_color_variation(self.decider.board)
                self.decider.sellTroop(loc)
                self.sell = False
                self.start = False
            elif self.start:
                if self.cardLimit < 6:
                    self.cardLimit += 1
                reward = self._get_reward_for_round(False)
                self.start = False

            print(f"Agent took action: {action}")
            self.start = False
            self.actions_this_turn += 1

            # Add logic to execute the play/merge action here.
            pos = action % 18
            cardPos = (action // 18)
            card = self.current_state["hand_cards"][cardPos]

            if self.board_state[pos] != 0 and self.board_state[pos] != card:
                reward -= 5
            elif self.current_state["elixir"] < self.costs[card]:
                reward -= 5
            elif (not (card in self.board_state)) and ((self.curCards < 6 and self.curCards >= self.cardLimit) or self.current_state["elixir"] < self.costs[card]):
                reward -= 5
            else:
                # encourage merging
                if card in self.board_state:
                    reward += 5
                elif self.curCards < 7:
                    self.board_state[pos] = card
                    self.curCards += 1
                    for trait in self.card_traits[card]:
                        self.combo_counts[trait] += 1
                        if trait in self.threes and self.combo_counts[trait] == 3:
                            reward += 20
                        elif self.combo_counts[trait] == 2:
                            reward += 10
                        elif self.combo_counts[trait] == 4:
                            reward += 30

                self.mouse.drag_card(self.coors[cardPos][0], self.coors[cardPos][1], self.board[pos][0], self.board[pos][1])
            
        # After the action, you would check the game state again.
        observation = self._get_obs()
        
        self.current_state = observation
        info = {}

        if self.filename:
            self.photographer.deletePicture(self.filename)
        
        print(f"Reward: {reward}")
        return observation, reward, terminated, False, info

    def _get_reward_for_round(self, over):
        """
        Calculates the reward based on the current action and observation.
        """
        # A very basic, conceptual reward function.
        # You'll need to expand on this significantly.
        
        # Placeholder for your vision logic to get the game result.
        
        if over:
            player_rank = self.vision.findRank()
            # Assign a reward based on the rank.
            if player_rank == 1:
                return 200 # Huge reward for winning
            elif player_rank == 2:
                return 100 # Decent reward for second place
            elif player_rank == 3:
                return -100 # Small penalty for third
            else:
                return -200 # Heavy penalty for fourth/loss
        else:
            new_health = self.vision.findHealth()
            health_change = self.current_health - new_health
            if health_change > 0:
                self.current_health = new_health
                return -10 * health_change
            else:
                self.current_health = new_health
                return 30

    def render(self, mode="human"):
        """
        Optional: Displays the current game screen.
        """
        if mode == "human":
            # You would implement this to show the game screen in a window for debugging.
            # For a headless environment, this is not necessary.
            pass

    def close(self):
        """
        Cleans up the environment, if necessary.
        """
        pass
