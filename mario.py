# Import the necessary packages
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import time

# Initialize the Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='console', apply_api_compatibility=True)

# Limit the action space to walk right or jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

# Create a variable to store the number of episodes
num_episodes = 10

# Create a list to store the total rewards for each episode
total_rewards = []

# Create a list to store the time taken for each episode
time_taken = []

# Use a for loop to iterate over the number of episodes
for i in range(num_episodes):
    # Record the start time of the current episode
    start_time = time.time()

    # Reset the environment to start a new episode
    state = env.reset()

    # Create a variable to store the cumulative reward for the current episode
    episode_reward = 0

    # Loop until the episode is done
    done = False
    while not done:
        # Choose a random action from the action space
        action = env.action_space.sample()

        # Take the action and observe the next state, reward, done flag, and info
        output = env.step(action)
        next_state = output[0]
        reward = output[1]
        done = output[2]
        info = output[3]

        # Update the episode reward
        episode_reward += reward

    # Record the end time of the current episode and calculate the time difference
    end_time = time.time()
    episode_time = end_time - start_time

    # Append the episode reward to the total rewards list
    total_rewards.append(episode_reward)

    # Append the episode time to the time taken list
    time_taken.append(episode_time)

    # Calculate the average time taken per episode
    avg_time = np.mean(time_taken)

    # Calculate the estimate of the remaining time by multiplying the average time by 
    #the number of episodes left
    remaining_time = avg_time * (num_episodes - i - 1)

    # Format the time values to show only one decimal place
    episode_time = round(episode_time, 1)
    remaining_time = round(remaining_time, 1)

    # Print the current episode number and 
    #the total number of episodes along with 
    #the time taken and 
    #the estimate of remaining time 
    print(f"Episode {i+1} of {num_episodes} | Time taken: {episode_time} seconds | Estimate of remaining time: {remaining_time} seconds")

# Close the environment when finished
env.close()

# Calculate some statistics from 
#the total rewards list and 
#the total times list 
avg_reward = np.mean(total_rewards)
min_reward = np.min(total_rewards)
max_reward = np.max(total_rewards)
std_reward = np.std(total_rewards)
avg_time = round(avg_time, 1)
total_time = round(sum(time_taken), 1)

# Print the statistics to show the performance of the agent
print(f"Average reward: {avg_reward}")
print(f"Minimum reward: {min_reward}")
print(f"Maximum reward: {max_reward}")
print(f"Standard deviation of reward: {std_reward}")
print(f"Average time per episode: {avg_time} seconds")
print(f"Total time: {total_time} seconds")
