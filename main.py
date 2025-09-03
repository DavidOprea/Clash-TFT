import picture
import vision
import mouse_control
import strategizer
import time
import os
import cv2
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import time

from ClashMergeEnv import ClashMergeEnv

def get_next_checkpoint_prefix(base_path: str, prefix: str) -> str:
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return f"{prefix}_1"

    all_files = os.listdir(base_path)

    checkpoint_files = [f for f in all_files if f.startswith(prefix)]

    next_checkpoint_number = len(checkpoint_files) + 1

    return f"{prefix}_{next_checkpoint_number}"

def main():
    """
    Main training function for the reinforcement learning agent.
    This script will instantiate the environment, train the agent, save the model,
    and then evaluate its performance.

    Gemini really helped with this stuff
    """
    # 1. Create the custom environment
    model_path = "checkpoints/ppo_clash_merge_3_4000_steps.zip"

    def make_env():
        env = ClashMergeEnv()
        LOG_DIR = "./logs/"
        env = Monitor(env, LOG_DIR)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    print("Environment created. Starting training.")
    
    # 2. Define and instantiate the RL model
    
    print(f"Loading model from '{model_path}'...")
    model = PPO.load(model_path, env=env, device="cpu")
    print("Model loaded successfully. Continuing training...")
    

    # 3. Start the training process
    total_timesteps = 100000 # I can't really do millions right now
    print(f"Training for {total_timesteps} timesteps...")

    new_prefix = get_next_checkpoint_prefix('./checkpoints/', 'ppo_clash_merge')
    
    # Optional: Save a checkpoint every 10,000 timesteps to avoid losing progress
    checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./checkpoints/',
                                             name_prefix=new_prefix)
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    end_time = time.time()
    
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # 4. Save the final trained model
    model_path = "clash_merge_ppo_model"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    # 5. Evaluate the trained agent's performance
    print("Evaluating trained policy...")
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

# Previous function I used to set everything up before making the model
def maiinn():
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
            detect.findBattle()
        
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

if __name__ == '__main__':
    main()