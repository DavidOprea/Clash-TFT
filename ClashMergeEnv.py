import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from vision import Vision
import mouse_control

class ClashMergeEnv(gym.Env):
    """
    A custom environment for the Clash Merge Tactics game that follows the gym interface.
    This class connects the game's state to a reinforcement learning algorithm.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self):
        super(ClashMergeEnv, self).__init__()
        
        # Initialize the Vision class from your existing code.
        self.vision = Vision()

        # Initialize the mouse control
        self.mouse = mouse_control.Mouse()

        # Initialize board state
        self.board_state = np.zeros((5, 4), dtype=np.int8) 
        self.coors = [[120, 701], [180,701], [240,701]]
        self.board = [[135, 440], [180, 440], [225, 440], [270, 440], [315, 440], 
                      [155, 470], [200, 470], [245, 470], [290, 470], [335, 470], 
                      [135, 500], [180, 500], [225, 500], [270, 500], [315, 500],
                      [175, 530], [200, 530], [245, 530], [290, 530], [335, 530]]
        
        # Define the state space (what the agent sees).
        # We'll use a simple multi-dimensional array for now.
        # This will contain elixir count, card state, and a board representation.
        # The numbers here are placeholders; you'll need to determine the correct dimensions.
        self.observation_space = spaces.Dict({
            "elixir": spaces.Box(0, 20, shape=(1,), dtype=np.float32),
            "hand_cards": spaces.MultiDiscrete([20, 20, 20]),  # Assuming 20 unique cards
            "board_state": self.board_state # Assuming a 5x4 grid
        })

        # --- UPDATED: Action Space with "Do Nothing" Action ---
        # We now have a larger discrete space to include the "wait" action.
        # The total number of play card actions (5 * 4 * 10) plus one for "do nothing".
        self.total_card_actions = 5 * 4 * 20
        self.action_space = spaces.Discrete(self.total_card_actions + 1)

        self.game_over = False
        self.current_state = None

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
        elixir_img, cardCoors = self.vision.findTemps()
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
        
        self.game_over = False
        observation = self._get_obs()
        info = {}
        self.board_state = np.zeros((5, 4), dtype=np.int8) 
        return observation, info

    def step(self, action):
        """
        Performs an action in the environment and returns the new state, reward, and other info.
        """
        if self.game_over:
            # If the game is over, the agent can't do anything.
            return self.current_state, 0, True, False, {}
        
        # --- UPDATED: Handle the "Do Nothing" action ---
        if action == self.total_card_actions:
            print("Agent chose to end turn.")
            # This is where the game would advance to the next round.
            # We calculate and return the total reward for the round.
            # reward = self._get_reward_for_round() - I'LL DO THIS IN MAIN PROBABLY
            terminated = self.game_over # Check for game-over conditions.
            
            # Reset counters for the next turn.
            self.actions_this_turn = 0
        else:
            # Here's where you would translate the action index into a game command.
            # For example, if action = 10, you might play the second card at the third location.
            # You would use your mouse_drag_script to perform the action.
            print(f"Agent took action: {action}")
            reward = 0.1
            self.actions_this_turn += 1
            self.turn_reward += reward

            # Add logic to execute the play/merge action here.
            pos = action % 20
            y = (action % 20) // 5
            x = (action % 20) % 5
            card = (action // 20) + 1

            if self.board_state[y][x] != 0 and self.board_state[y][x] != card:
                action = self.total_card_actions
            else:
                for i in range(3):
                    if self.observation_space["hand_cards"][i] == card:
                        self.mouse.drag_card(self.coors[0], self.coors[1], self.board[pos][0], self.board[pos][1])
            
        # After the action, you would check the game state again.
        observation = self._get_obs()
        
        self.current_state = observation
        info = {}
        
        return observation, reward, terminated, False, info

    def _get_reward_for_round(self, action, observation):
        """
        Calculates the reward based on the current action and observation.
        """
        # A very basic, conceptual reward function.
        # You'll need to expand on this significantly.
        
        # Placeholder for your vision logic to get the game result.
        total = 0
        
        if self.game_over:
            player_rank = self.vision.findRank()
            # Assign a reward based on the rank.
            if player_rank == 1:
                total += 75 # Huge reward for winning
            elif player_rank == 2:
                total += 25 # Decent reward for second place
            elif player_rank == 3:
                total -= 25 # Small penalty for third
            else:
                total -= 75 # Heavy penalty for fourth/loss
        
        self.turn_reward = 0
        return total

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
