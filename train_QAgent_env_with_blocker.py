import copy
import os
import pickle
from typing import List

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from agents.DDQNagent import DDQNAgent
from agents.DDQNagent_prioritized import DDQNAgentPrioritized
from engine.game_rule import random_game_spec
from env import (
    discretize_state_blocker,
    discretize_state_runner,
    discretize_state_runner_solo,
    envBlocker,
)

GameHistory = List[List]
load_dotenv()


def load_agent(pretrained_model, agent_cfg):
    agent = DDQNAgent(
        model_type=agent_cfg.get("model_type", "dueling"),  # Default to 'dueling' if not specified
        state_dim=agent_cfg.get("state_dim", 8),  # Default to 8 if not specified
        action_dim=agent_cfg.get("action_dim", 7),  # Default to 6 if not specified
        epsilon=agent_cfg.get("epsilon", 0.9),  # Default to 0.9 if not specified
        epsilon_min=agent_cfg.get("epsilon_min", 0.05),  # Default to 0.05 if not specified
        epsilon_decay=agent_cfg.get("epsilon_decay", 0.995),  # Default to 0.995 if not specified
        gamma=agent_cfg.get("gamma", 0.9),  # Default to 0.9 if not specified
        alpha=agent_cfg.get("alpha", 1e-4),  # Default to 1e-4 if not specified
        batch_size=agent_cfg.get("batch_size", 64),  # Default to 64 if not specified
        replay_buffer_size=agent_cfg.get("replay_buffer_size", 2**18),  # Default to 2^18 if not specified
    )

    if os.path.exists(pretrained_model):
        print(f"Loading pre-trained model from {pretrained_model}")
        agent.load(pretrained_model)
    else:
        print("No pre-trained model found. Initializing a new agent.")

    return agent


# Ensure that the replay directory exists
# replays_directory = "RunnerWithBlockerTraining"
replays_directory = "BlockerWithRunnerTraining"
replays_directory_prefix = f"replays/{replays_directory}"
os.makedirs(replays_directory_prefix, exist_ok=True)
os.makedirs("checkpoints_training", exist_ok=True)
os.makedirs("models_weights", exist_ok=True)

agent_to_train_idx = 1  # Index of the agent we are training (0 for blocker, 1 for runner)
epochs = 4_000_000

# blocker_model_name = "blocker_trained_on_duel_solo_runner.pkl"
blocker_model_name = "blocker_based_on_duel_solo_runner.pkl"
blocker_model_path = f"./models_weights/{blocker_model_name}"

runner_model_name = "pliska_be_good.pkl"
runner_model_path = "checkpoints_training/checkpoint_3800000_pliska_be_good.pkl"
# runner_model_path = f"./models_weights/{runner_model_name}"

out_model_name = blocker_model_name if agent_to_train_idx == 0 else runner_model_name
out_model_path = f"./models_weights/{out_model_name}"

wandbDebug = False  # Set to True to enable wandb logging

training_cfg = {
    "epochs": epochs,
    "start_learning_at": 2000,
    "learn_per": 3,
    "eval_per": 10000,
    "debug_per": -1,
    "update_target_every": 500,
    "log_per": 10,
    "save_model_per": 100000,
}

blocker_cfg = {
    "model_type": "dueling",  # Model type for the agent, can be 'standard' or 'dueling'
    "state_dim": 19,  # State dimension for the agent
    "action_dim": 7,  # Action dimension for the agent
    "epsilon": 0.01 if agent_to_train_idx == 1 else 0.9,  # Initial epsilon value
    "epsilon_min": 0.01,  # Minimum epsilon value
    "epsilon_decay": 0.9999,  # Decay factor for epsilon
    "batch_size": 64,  # Batch size for training
    "alpha": 1e-4,  # Learning rate for the optimizer
    "gamma": 0.9,  # Discount factor for future rewards
    "replay_buffer_size": 2**18,  # Size of the replay buffer
}

runner_cfg = {
    "model_type": "dueling",  # Model type for the agent, can be 'standard' or 'dueling'
    "state_dim": 13,  # State dimension for the agent
    "action_dim": 7,  # Action dimension for the agent
    "epsilon": 0.01 if agent_to_train_idx == 0 else 0.9,  # Initial epsilon value
    "epsilon_min": 0.01,  # Minimum epsilon value
    "epsilon_decay": 0.9999,  # Decay factor for epsilon
    "batch_size": 64,  # Batch size for training
    "alpha": 1e-4,  # Learning rate for the optimizer
    "gamma": 0.9,  # Discount factor for future rewards
    "replay_buffer_size": 2**18,  # Size of the replay buffer
}

# Load player and enemy agents
blocker_agent = load_agent(blocker_model_path, blocker_cfg)
runner_agent = load_agent(runner_model_path, runner_cfg)

agents = [
    blocker_agent,
    runner_agent,
]  # List of agents in the environment  # Index of the agent we are training (blocker agent)
agent_to_train = agents[agent_to_train_idx]

# Game variables
sum_history_length, timeouts = 0, 0
num_of_games = 0
history: GameHistory = []

# Initialize environment with the correct number of agents (2 players + 2 enemies for 2v2)
num_agents = 2  # 1 player + 1 enemy
env = envBlocker(num_agents=num_agents)

game_spec = random_game_spec(pod_count=2, lap_count=3)  # Two pods: player and enemy
current_states = env.reset(game_spec)


if wandbDebug:
    wandb.login(key=os.getenv("WANDB_API_KEY"))

if wandbDebug:
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"MadPodRacing",
        # track hyperparameters and run metadata,
        name=out_model_name,
        config={
            "Method": out_model_name,
        }
        | training_cfg
        | blocker_cfg
        | runner_cfg,
        # do not save the code
        save_code=False,
    )

for epoch in tqdm(range(training_cfg["epochs"])):
    history.append(copy.deepcopy(env.states_for_history))

    # Split current_state into player and enemy states
    blocker_state = current_states[0]  # First state for the player
    runner_state = current_states[1]  # Remaining state for the enemy

    # Blocker's turn
    blocker_discretized_state = discretize_state_blocker(blocker_state, runner_state)
    blocker_action_idx = blocker_agent.act_greedy(blocker_discretized_state)

    # Enemy's turn
    runner_discretized_state = discretize_state_runner(runner_state, blocker_state)
    # runner_discretized_state = discretize_state_runner_solo(runner_state)
    runner_action_idx = runner_agent.act_greedy(runner_discretized_state)

    # Step environment with actions for all agents
    rewards, new_states, done = env.step([blocker_action_idx, runner_action_idx])

    blocker_reward = rewards[0]  # Reward for the player
    runner_reward = rewards[1]  # Reward for the enemy

    if agent_to_train_idx == 0:
        blocker_agent.replay_buffer.push(
            blocker_discretized_state,
            blocker_action_idx,
            blocker_reward,
            discretize_state_blocker(new_states[0], new_states[1]),
            done,
        )
    else:
        runner_agent.replay_buffer.push(
            runner_discretized_state,
            runner_action_idx,
            runner_reward,
            discretize_state_runner(new_states[1], new_states[0]),
            done,
        )

    current_states = new_states

    if epoch >= training_cfg["start_learning_at"] and epoch % training_cfg["learn_per"] == 0:
        agent_to_train.learn()
        agent_to_train.decay_epsilon()

    if epoch >= training_cfg["start_learning_at"] and epoch % training_cfg["update_target_every"] == 0:
        agent_to_train.update_target_model()

    if training_cfg["eval_per"] != -1 and epoch % training_cfg["eval_per"] == 0:
        agent_to_train.evaluate()
        agent_to_train.train()

    if training_cfg["debug_per"] != -1 and epoch % training_cfg["debug_per"] == 0:
        print(f"[{epoch + 1}] checkpoints left: {[len(p.pod_next_checkpoints) for p in history[-1]]}")
        print(f"History length: {len(history)}")
        print(f"Number of games: {num_of_games + 1}")
        print(f"Epsilon: {agent_to_train.epsilon:.3f}")
        print(f"Replay buffer size: {len(agent_to_train.replay_buffer)}")
        print(f"Average turns: {sum_history_length / (num_of_games+1):.2f}")
        print(f"Timeouts: {timeouts}")

    if training_cfg["save_model_per"] != -1 and (epoch + 1) % training_cfg["save_model_per"] == 0:
        agent_to_train.save(f"./checkpoints_training/checkpoint_{epoch+1}_{out_model_name}")
        print(f"Model saved at epoch {epoch+1} to ./checkpoints_training/checkpoint_{epoch+1}_{out_model_name}")

    if training_cfg["log_per"] != -1 and epoch % training_cfg["log_per"] == 0:
        if wandbDebug:
            log_info = {
                "Epoch": epoch,
                "Epsilon": agent_to_train.epsilon,
                "Replay Buffer Size": len(agent_to_train.replay_buffer),
                "Average Turns": sum_history_length / (num_of_games + 1),
                "Timeouts": timeouts,
                "History Length": len(history),
            }
            # print(f"Dodano log: {log_info} do WandB")
            wandb.log(log_info)

    # Save replay after each game
    if done:
        if env.runner_timeout > 0:
            print(
                f"Epoch {epoch}: Game Over. Pod has completed all laps. After {len(history)} turns. Current Epsilon: {agent_to_train.epsilon:.3f}, enemy timeout: {env.runner_timeout}"
            )
        else:
            print(
                f"Epoch {epoch}: Game Over. Pod has timed out. After {len(history)} turns. Current Epsilon: {agent_to_train.epsilon:.3f}"
            )
            timeouts += 1

        history.append(copy.deepcopy([new_states[0], new_states[1]]))

        with open(f"{replays_directory_prefix}/replay_{epoch}.pkl", "wb") as f:
            pickle.dump((("Blocker", "Runner"), game_spec, history), f)

        if wandbDebug and env.timeout != 0:
            wandb.log(
                {
                    "Game Length": len(history),
                    "Epsilon": agent_to_train.epsilon,
                    "Average Turns": sum_history_length / (num_of_games + 1),
                }
            )

        num_of_games += 1
        sum_history_length += len(history)
        game_spec = random_game_spec(pod_count=2, lap_count=3, small_map=False)
        current_state = env.reset(game_spec)
        history: GameHistory = []
        agent_to_train.init()
        continue

# Save the final model to the specified output path
agent_to_train.save(out_model_path)
print(f"Training completed. Final model saved to {out_model_path}.")
