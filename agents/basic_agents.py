import math
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from engine.game_rule import GameSpec
from engine.game_sim import Boost, Normal, PodMovement, Shield
from engine.util import game_world_size, get_min_rep_len, rad_to_deg
from engine.vec2 import Vec2


# Base Player interfaces
class Player(ABC):
    @abstractmethod
    def init(self) -> "Player":
        pass

    @abstractmethod
    def run(self, player_in: Tuple[List[Any], List[Any]]) -> Tuple[List[PodMovement], "Player"]:
        pass


class PlayerIO(ABC):
    @abstractmethod
    def init_io(self) -> "PlayerIO":
        pass

    @abstractmethod
    def run_io(self, player_in: Tuple[List[Any], List[Any]]) -> Tuple[List[PodMovement], "PlayerIO"]:
        pass


class WrapIO(PlayerIO):
    def __init__(self, player: Player) -> None:
        self.player = player

    def init_io(self) -> "WrapIO":
        return WrapIO(self.player.init())

    def run_io(self, player_in: Tuple[List[Any], List[Any]]) -> Tuple[List[PodMovement], "WrapIO"]:
        out, next_p = self.player.run(player_in)
        return out, WrapIO(next_p)


# Default and Elementary players
enable_search = False  # set to True to include search player


class DefaultPlayer(Player):
    def init(self) -> "DefaultPlayer":
        return self

    def run(self, player_in) -> Tuple[List[PodMovement], "DefaultPlayer"]:
        # Always move towards center - only one pod now
        target = Vec2(8000, 4500)
        mov = PodMovement(target, Normal(100))
        return [mov], self


class ElementaryPlayer(Player):
    def init(self) -> "ElementaryPlayer":
        return self

    def run(self, player_in) -> Tuple[List[PodMovement], "ElementaryPlayer"]:
        self_pods, _ = player_in
        movements = []
        for ps in self_pods:
            target = ps.pod_next_checkpoints[0] if ps.pod_next_checkpoints else Vec2(0, 0)
            movements.append(PodMovement(target, Normal(100)))
        return movements, self


# Process-based player
def new_process(cmd: str) -> "ProcessPlayer":
    return ProcessPlayer(cmd)


class ProcessPlayer(PlayerIO):
    def __init__(self, cmd: str) -> None:
        self.cmd = cmd
        self.proc = None  # type: Optional[subprocess.Popen]
        self.checkpoints = []

    def init_io(self) -> "ProcessPlayer":
        # start subprocess
        self.proc = subprocess.Popen(
            self.cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self

    def run_io(self, player_in) -> Tuple[List[PodMovement], "ProcessPlayer"]:
        self_pods, oppo_pods = player_in
        # On first call, send map info
        if self.checkpoints == []:
            ckpt_list = player_in[0][0].pod_next_checkpoints
            count = get_min_rep_len(ckpt_list)
            laps = len(ckpt_list) // count
            # send laps and checkpoint count
            print(laps, file=self.proc.stdin)
            print(count, file=self.proc.stdin)
            for ck in ckpt_list[:count]:
                print(f"{round(ck.x)} {round(ck.y)}", file=self.proc.stdin)
            self.proc.stdin.flush()
            self.checkpoints = ckpt_list[:count]
        # send pod states
        for ps in self_pods + oppo_pods:
            idx = self.checkpoints.index(ps.pod_next_checkpoints[0])
            ang = round(rad_to_deg(ps.pod_angle or 0))
            line = f"{round(ps.pod_position.x)} {round(ps.pod_position.y)} {round(ps.pod_speed.x)} {round(ps.pod_speed.y)} {ang} {idx}"
            print(line, file=self.proc.stdin)
        self.proc.stdin.flush()
        # read movements
        outs = []
        for _ in self_pods:
            line = self.proc.stdout.readline().split()
            tx, ty, thr = line
            target = Vec2(float(tx), float(ty))
            if thr == "BOOST":
                thrust = Boost()
            elif thr == "SHIELD":
                thrust = Shield()
            else:
                thrust = Normal(int(thr))
            outs.append(PodMovement(target, thrust))
        return outs, self


class RandomPlayer(Player):
    def init(self) -> "RandomPlayer":
        return self

    def run(self, player_in) -> Tuple[List[PodMovement], "RandomPlayer"]:
        pods, _ = player_in
        outs = []
        for _ in pods:
            tx = random.uniform(0, game_world_size.x)
            ty = random.uniform(0, game_world_size.y)
            thrust_value = random.randint(0, 100)
            action = random.choice([Boost(), Shield(), Normal(thrust_value)])
            outs.append(PodMovement(Vec2(tx, ty), action))
        return outs, self
