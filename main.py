#!/usr/bin/env python3
import argparse
import pickle
import sys
from typing import Any, List, Optional, Tuple

from agents.basic_agents import ElementaryPlayer, RandomPlayer, WrapIO
from engine.game_rule import (
    game_end,
    game_end_solo,
    game_end_solo_v2,
    random_game_spec,
    run_game,
    run_game_one_player,
)
from engine.interact import game_animate
from engine.interact import winner as compute_winner

sys.setrecursionlimit(2137)


class GameConfig:
    def __init__(
        self,
        player1_mode: Tuple[str, Any, str],
        player2_mode: Tuple[str, Any, str],
        show_animation: bool = True,
        save_history: Optional[str] = None,
        play_file: Optional[str] = None,
    ) -> None:
        self.player1_mode = player1_mode
        self.player2_mode = player2_mode
        self.show_animation = show_animation
        self.save_history = save_history
        self.play_file = play_file

    @staticmethod
    def default() -> "GameConfig":
        return GameConfig(("default", 1, "Boss1"), ("default", 0, "Boss0"))


def parse_player_mode(arg: str, name: str) -> Tuple[str, Any, str]:
    if arg.isdigit():
        return ("default", int(arg), name)
    elif arg == "io":
        return ("io", None, name)
    else:
        return ("exec", arg, name)


def parse_args(argv: List[str]) -> GameConfig:
    parser = argparse.ArgumentParser(description="csb-evolved")
    parser.add_argument("-p1", nargs=2, metavar=("PLAYER", "NAME"))
    parser.add_argument("-p2", nargs=2, metavar=("PLAYER", "NAME"))
    parser.add_argument("-noAnimation", action="store_true")
    parser.add_argument("-saveGame", metavar="PATH")
    parser.add_argument("-playFile", metavar="PATH")
    args = parser.parse_args(argv)

    cfg = GameConfig.default()
    if args.p1:
        mode, name = args.p1
        cfg.player1_mode = parse_player_mode(mode, name)
    if args.p2:
        mode, name = args.p2
        cfg.player2_mode = parse_player_mode(mode, name)
    if args.noAnimation:
        cfg.show_animation = False
    if args.saveGame:
        cfg.save_history = args.saveGame
    if args.playFile:
        cfg.play_file = args.playFile
    return cfg


def main_solo_player() -> None:
    cfg = parse_args(sys.argv[1:])

    if cfg.play_file:
        with open(cfg.play_file, "rb") as f:
            (name1,), spec, history = pickle.load(f)
        game_animate([name1], turn_per_sec=4.5, spec=spec, history=history)
        sys.exit(0)

    player1_io = ElementaryPlayer()
    name1 = "Random1"

    spec = random_game_spec(pod_count=1, lap_count=1, small_map=True)
    history = run_game_one_player(player1_io, spec, game_end)

    print(f"Game history: {history}")
    print(f"Game finished. Winner: {compute_winner(history)}")

    game_animate([name1], turn_per_sec=6.0, spec=spec, history=history)


def main() -> None:
    cfg = parse_args(sys.argv[1:])

    # If replaying a saved game:
    if cfg.play_file:
        with open(cfg.play_file, "rb") as f:
            (name1, name2), spec, history = pickle.load(f)
        game_animate((name1, name2), turn_per_sec=4.5, spec=spec, history=history)
        sys.exit(0)

    # Otherwise, run a new game:
    # 1. Create players
    player1_io = ElementaryPlayer()
    player2_io = ElementaryPlayer()
    name1 = "Random1"
    name2 = "Random2"

    # 2. Generate a random GameSpec and run the simulation
    spec = random_game_spec(pod_count=2, lap_count=3, small_map=True)
    history = run_game((player1_io, player2_io), spec, game_end)

    # 3. Optionally save history
    if cfg.save_history:
        with open(cfg.save_history, "wb") as f:
            pickle.dump(((name1, name2), spec, history), f)

    # 4. Animate or print the winner
    if cfg.show_animation:
        game_animate([name1, name2], turn_per_sec=6.0, spec=spec, history=history)
    else:
        # compute_winner returns 1 or 2
        print(compute_winner(history))


if __name__ == "__main__":
    main()
    # main_solo_player()
