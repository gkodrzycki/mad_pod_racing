import copy
import math
import os
import pickle
import random
from random import randint
from typing import List

import numpy as np
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from agents.DDQNagent import DDQNAgent
from agents.DDQNagent_prioritized import DDQNAgentPrioritized
from engine.game_rule import (
    random_game_spec,
    specific_game_spec,
    specific_game_spec_w_randomly_positioned_ckpts,
)
from engine.game_sim import Boost, Normal, PodState
from engine.vec2 import Vec2  # zakładamy, że masz tę klasę
from env import discretize_state_runner, discretize_state_runner_solo, envSolo

GameHistory = List[List[PodState]]
load_dotenv()


def test_agent(agent):

    truth_set_ins = [
        [0.47135262, 0.99999178, -0.00405385, 0.551, 0.0, -0.52936071, 0.84839686, 0.36654079],
        [0.438748, 0.99999153, -0.00411542, 0.553, 0.0, -0.52930848, 0.84842945, 0.36654079],
        [0.40607661, 0.99999101, -0.00423961, 0.554, -0.001, -0.5292031, 0.84849518, 0.36654079],
        [0.37324614, 0.99988272, -0.01531512, 0.647, 0.035, -0.51977282, 0.85430452, 0.36654079],
        [0.33587597, 0.99994238, -0.01073516, 0.634, 0.032, -0.52368038, 0.85191482, 0.36654079],
        [0.35295688, 0.99988952, -0.01486448, 0.422, 0.081, -0.75862166, 0.65153141, 0.36654079],
        [0.10724854, 0.99625713, -0.08643922, 0.286, 0.041, -0.5525616, -0.83347206, 0.38851144],
        [0.08796043, 0.99738865, -0.07222106, 0.326, 0.033, -0.54061779, -0.84126833, 0.38851144],
    ]

    truth_set_outs = [3, 3, 3, 3, 3, 3, 4, 3]

    correct = 0
    for input_state, expected_action in zip(truth_set_ins, truth_set_outs):
        action_idx = agent.act_non_greedy(input_state)
        correct += action_idx == expected_action

    return correct / len(truth_set_ins) * 100


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


env = envSolo()


# Ensure that the training directory exists
replays_directory = "new_bot"
replays_directory_prefix = "replays/" + replays_directory

os.makedirs(replays_directory_prefix, exist_ok=True)
os.makedirs("checkpoints_training", exist_ok=True)
os.makedirs("models_weights", exist_ok=True)

pretrained_model_name = "simplified_runner.pkl"
pretrained_model = f"./models_weights/{pretrained_model_name}"

out_model_name = pretrained_model_name
out_model = f"./models_weights/{out_model_name}"

wandbDebug = False  # Set to True to enable wandb logging

epochs = 4_000_000

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

runner_cfg = {
    "model_type": "dueling",  # Model type for the agent, can be 'standard' or 'dueling'
    "state_dim": 8,  # State dimension for the agent
    "action_dim": 6,  # Action dimension for the agent
    "epsilon": 0.9,  # Initial epsilon value
    "epsilon_min": 0.01,  # Minimum epsilon value
    "epsilon_decay": 0.9999,  # Decay factor for epsilon
    "batch_size": 64,  # Batch size for training
    "alpha": 1e-4,  # Learning rate for the optimizer
    "gamma": 0.9,  # Discount factor for future rewards
    "replay_buffer_size": 2**18,  # Size of the replay buffer
}

agent = load_agent(pretrained_model, runner_cfg)

sum_history_length, timeouts = 0, 0

num_of_games = 0

game_spec = random_game_spec(pod_count=1, lap_count=3)  # Use a random game spec for training
current_state = env.reset(game_spec)

agent = agent.init()
history: GameHistory = []

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
        | runner_cfg,
        # do not save the code
        save_code=False,
    )


for epoch in tqdm(range(training_cfg["epochs"])):

    history.append([copy.deepcopy(env.state_for_history)])

    discretized_state = discretize_state_runner_solo(current_state)
    action_idx = agent.act_greedy(discretized_state)

    reward, new_state, done = env.step(action_idx)

    # print(f"Epoch {epoch}: Action index: {action_idx}, Reward: {reward}, Done: {done}, State: {new_state}")

    agent.replay_buffer.push(discretized_state, action_idx, reward, discretize_state_runner_solo(new_state), done)

    current_state = new_state

    if epoch >= training_cfg["start_learning_at"] and epoch % training_cfg["learn_per"] == 0:
        agent.learn()
        agent.decay_epsilon()

    if epoch >= training_cfg["start_learning_at"] and epoch % training_cfg["update_target_every"] == 0:
        agent.update_target_model()

    if training_cfg["eval_per"] != -1 and epoch % training_cfg["eval_per"] == 0:
        agent.evaluate()
        print(f"Eval: {test_agent(agent)}")
        agent.train()

    if training_cfg["debug_per"] != -1 and epoch % training_cfg["debug_per"] == 0:
        print(f"[{epoch + 1}] checkpoints left: {[len(p.pod_next_checkpoints) for p in history[-1]]}")
        print(f"History length: {len(history)}")
        print(f"Number of games: {num_of_games + 1}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print(f"Replay buffer size: {len(agent.replay_buffer)}")
        print(f"Average turns: {sum_history_length / (num_of_games+1):.2f}")
        print(f"Timeouts: {timeouts}")
        agent.evaluate()
        print(f"Accuracy: {test_agent(agent):.2f}%")
        agent.train()

    if training_cfg["save_model_per"] != -1 and (epoch + 1) % training_cfg["save_model_per"] == 0:
        agent.save(f"./checkpoints_training/checkpoint_{epoch+1}_{out_model_name}")
        print(f"Model saved at epoch {epoch+1} to ./checkpoints_training/checkpoint_{epoch+1}_{out_model_name}")

    if training_cfg["log_per"] != -1 and epoch % training_cfg["log_per"] == 0:
        if wandbDebug:
            agent.evaluate()
            accuracy = test_agent(agent)
            agent.train()
            log_info = {
                "Epoch": epoch,
                "Epsilon": agent.epsilon,
                "Replay Buffer Size": len(agent.replay_buffer),
                "Average Turns": sum_history_length / (num_of_games + 1),
                "Timeouts": timeouts,
                "History Length": len(history),
                "Accuracy": accuracy,
            }
            # print(f"Dodano log: {log_info} do WandB")
            wandb.log(log_info)

    if done:
        if env.timeout > 0:
            print(
                f"Epoch {epoch}: Game Over. Pod has completed all laps. After {len(history)} turns. Current Epsilon: {agent.epsilon:.3f}"
            )
        else:
            # print(f"Epoch {epoch}: Game Over. Pod has timed out. After {len(history)} turns. Current Epsilon: {agent.epsilon:.3f}")
            ...

        with open(f"replays/{replays_directory}/replay_{epoch}.pkl", "wb") as f:
            pickle.dump((("DQN",), game_spec, history), f)

        timeouts += 1 if env.timeout == 0 else 0
        num_of_games += 1
        sum_history_length += len(history)

        if wandbDebug and env.timeout != 0:
            wandb.log(
                {
                    "Game Length": len(history),
                    "Timeout": env.timeout == 0,
                    "Epsilon": agent.epsilon,
                    "Average Turns": sum_history_length / (num_of_games + 1),
                }
            )

        game_spec = random_game_spec(pod_count=1, lap_count=3)
        current_state = env.reset(game_spec)
        history: GameHistory = []
        agent.init()
        continue

agent.save(f"{out_model}")
print(f"Training completed. Final model saved to {out_model}")

print(f"Out of {num_of_games} games, {timeouts} ended with timeout.")
with open(f"replays/{replays_directory}/replay_end.pkl", "wb") as f:
    pickle.dump((("DQN",), game_spec, history), f)
