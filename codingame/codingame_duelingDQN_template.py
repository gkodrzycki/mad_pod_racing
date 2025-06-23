import math
import sys

import numpy as np


def decode_unicode_weights(encoded_str, shape, scale_factor=512.0, offset=12.0):
    weights = []
    for c in encoded_str:
        code = ord(c)
        if code >= 0x8000:
            code -= 0x10000
        weights.append(code / scale_factor - offset)
    return np.array(weights, dtype=np.float32).reshape(shape)


scale_factor = float("<scale_factor>")
offset_unicode = float("<offset>")

runner_input = int("<runner_input>")
blocker_input = int("<blocker_input>")

runner_output = int("<runner_output>")
blocker_output = int("<blocker_output>")

all_weights_runner = "<all_weights_runner>"
all_weights_blocker = "<all_weights_blocker>"


layer_shapes = [
    (64, runner_input),  # shared_fc1_weight
    (64,),  # shared_fc1_bias
    (64, 64),  # shared_fc2_weight
    (64,),  # shared_fc2_bias
    (32, 64),  # value_fc1_weight
    (32,),  # value_fc1_bias
    (1, 32),  # value_fc2_weight
    (1,),  # value_fc2_bias
    (32, 64),  # adv_fc1_weight
    (32,),  # adv_fc1_bias
    (runner_output, 32),  # adv_fc2_weight
    (runner_output,),  # adv_fc2_bias
]

counts = [np.prod(shape) for shape in layer_shapes]
offset = 0
decoded_layers_runner = []
for cnt, shape in zip(counts, layer_shapes):
    chunk = all_weights_runner[offset : offset + cnt]
    decoded = decode_unicode_weights(chunk, shape)
    decoded_layers_runner.append(decoded)
    offset += cnt


class DuelingQNetworkRunner:
    def relu(self, x):
        return [max(0.0, val) for val in x]

    def __init__(self):
        self.w = decoded_layers_runner
        (
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.val_fc1_weight,
            self.val_fc1_bias,
            self.val_fc2_weight,
            self.val_fc2_bias,
            self.adv_fc1_weight,
            self.adv_fc1_bias,
            self.adv_fc2_weight,
            self.adv_fc2_bias,
        ) = self.w

    def forward_layer(self, x, weight, bias):
        out = []
        for i in range(len(weight)):
            val = bias[i]
            for j in range(len(x)):
                val += weight[i][j] * x[j]
            out.append(val)
        return out

    def forward(self, x):
        h1 = self.forward_layer(x, self.fc1_weight, self.fc1_bias)
        h1 = self.relu(h1)
        h2 = self.forward_layer(h1, self.fc2_weight, self.fc2_bias)
        h2 = self.relu(h2)
        v1 = self.forward_layer(h2, self.val_fc1_weight, self.val_fc1_bias)
        v1 = self.relu(v1)
        v = self.forward_layer(v1, self.val_fc2_weight, self.val_fc2_bias)[0]
        a1 = self.forward_layer(h2, self.adv_fc1_weight, self.adv_fc1_bias)
        a1 = self.relu(a1)
        a = self.forward_layer(a1, self.adv_fc2_weight, self.adv_fc2_bias)
        a_mean = sum(a) / len(a)
        q = [v + ai - a_mean for ai in a]
        return q


layer_shapes = [
    (64, blocker_input),  # shared_fc1_weight
    (64,),  # shared_fc1_bias
    (64, 64),  # shared_fc2_weight
    (64,),  # shared_fc2_bias
    (32, 64),  # value_fc1_weight
    (32,),  # value_fc1_bias
    (1, 32),  # value_fc2_weight
    (1,),  # value_fc2_bias
    (32, 64),  # adv_fc1_weight
    (32,),  # adv_fc1_bias
    (blocker_output, 32),  # adv_fc2_weight
    (blocker_output,),  # adv_fc2_bias
]

counts = [np.prod(shape) for shape in layer_shapes]
offset = 0
decoded_layers_blocker = []
for cnt, shape in zip(counts, layer_shapes):
    chunk = all_weights_blocker[offset : offset + cnt]
    decoded = decode_unicode_weights(chunk, shape)
    decoded_layers_blocker.append(decoded)
    offset += cnt


class DuelingQNetworkBlocker:
    def relu(self, x):
        return [max(0.0, val) for val in x]

    def __init__(self):
        self.w = decoded_layers_blocker
        (
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.val_fc1_weight,
            self.val_fc1_bias,
            self.val_fc2_weight,
            self.val_fc2_bias,
            self.adv_fc1_weight,
            self.adv_fc1_bias,
            self.adv_fc2_weight,
            self.adv_fc2_bias,
        ) = self.w

    def forward_layer(self, x, weight, bias):
        out = []
        for i in range(len(weight)):
            val = bias[i]
            for j in range(len(x)):
                val += weight[i][j] * x[j]
            out.append(val)
        return out

    def forward(self, x):
        h1 = self.forward_layer(x, self.fc1_weight, self.fc1_bias)
        h1 = self.relu(h1)
        h2 = self.forward_layer(h1, self.fc2_weight, self.fc2_bias)
        h2 = self.relu(h2)
        v1 = self.forward_layer(h2, self.val_fc1_weight, self.val_fc1_bias)
        v1 = self.relu(v1)
        v = self.forward_layer(v1, self.val_fc2_weight, self.val_fc2_bias)[0]
        a1 = self.forward_layer(h2, self.adv_fc1_weight, self.adv_fc1_bias)
        a1 = self.relu(a1)
        a = self.forward_layer(a1, self.adv_fc2_weight, self.adv_fc2_bias)
        a_mean = sum(a) / len(a)
        q = [v + ai - a_mean for ai in a]
        return q


runner_agent = DuelingQNetworkRunner()
blocker_agent = DuelingQNetworkBlocker()


class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def cross(self, p):
        return self.y * p.x - self.x * p.y

    def norm(self):
        temp = math.sqrt(self.x**2 + self.y**2)
        self.x /= temp
        self.y /= temp

    def angle_diff(self, p):
        return math.atan2(p.y - self.y, p.x - self.x)


def vec_by_angle(angle):
    return Vec(np.cos(angle), np.sin(angle))


def norm_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


checkpoints = []

laps = int(input())
checkpoint_count = int(input())

for i in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    checkpoints.append(Vec(checkpoint_x, checkpoint_y))


def get_input_runner(
    runner_x, runner_y, runner_vx, runner_vy, angle, next_cp_id, blocker_x, blocker_y, blocker_vx, blocker_vy
):
    cp1 = checkpoints[next_cp_id]
    cp2 = checkpoints[(next_cp_id + 1) % checkpoint_count]

    angle *= np.pi / 180

    runner_pos = Vec(runner_x, runner_y)
    runner_speed = Vec(runner_vx, runner_vy)
    blocker_pos = Vec(blocker_x, blocker_y)
    blocker_speed = Vec(blocker_vx, blocker_vy)

    distance = runner_pos.distance(cp1)
    angle_to_cp1 = runner_pos.angle_diff(cp1)

    angle -= angle_to_cp1

    angle_v = Vec(0, 0).angle_diff(runner_speed) - angle_to_cp1
    norm_v = math.sqrt(runner_vx**2 + runner_vy**2)

    angle_next_dir = cp1.angle_diff(cp2) - angle_to_cp1
    distance_cp1_cp2 = cp1.distance(cp2)

    angle_dot = vec_by_angle(angle).dot(Vec(1, 0))
    angle_cross = vec_by_angle(angle).cross(Vec(1, 0))

    v_dot = int(norm_v * vec_by_angle(angle_v).dot(Vec(1, 0)))
    v_cross = int(norm_v * vec_by_angle(angle_v).cross(Vec(1, 0)))

    next_dir_dot = vec_by_angle(angle_next_dir).dot(Vec(1, 0))
    next_dir_cross = vec_by_angle(angle_next_dir).cross(Vec(1, 0))

    distance_to_blocker = runner_pos.distance(blocker_pos) / 20000.0
    angle_to_blocker = runner_pos.angle_diff(blocker_pos)
    blocker_angle = angle - angle_to_blocker

    angle_to_blocker_dot = vec_by_angle(blocker_angle).dot(Vec(1, 0))
    angle_to_blocker_cross = vec_by_angle(blocker_angle).cross(Vec(1, 0))

    angle_v_blocker = Vec(0, 0).angle_diff(blocker_speed) - angle_to_blocker
    norm_v_blocker = math.sqrt(blocker_vx**2 + blocker_vy**2)

    blocker_v_dot = int(norm_v_blocker * vec_by_angle(angle_v_blocker).dot(Vec(1, 0))) / 1000.0
    blocker_v_cross = int(norm_v_blocker * vec_by_angle(angle_v_blocker).cross(Vec(1, 0))) / 1000.0

    return np.array(
        [
            (distance - 600.0) / 20000.0,
            angle_dot,
            angle_cross,
            v_dot / 1000.0,
            v_cross / 1000.0,
            next_dir_dot,
            next_dir_cross,
            (distance_cp1_cp2 - 1200.0) / 10000.0,
            distance_to_blocker,
            angle_to_blocker_dot,
            angle_to_blocker_cross,
            blocker_v_dot,
            blocker_v_cross,
        ]
    )


def get_input_blocker(blocker, enemy_to_block):
    b_x, b_y, b_vx, b_vy, b_angle, b_next_cp = blocker
    e_x, e_y, e_vx, e_vy, e_angle, e_next_cp = enemy_to_block

    b_speed = Vec(b_vx, b_vy)
    e_speed = Vec(e_vx, e_vy)

    CP1 = checkpoints[e_next_cp]

    CP11 = checkpoints[e_next_cp]
    CP22 = checkpoints[(e_next_cp + 1) % checkpoint_count]
    CP33 = checkpoints[(e_next_cp + 2) % checkpoint_count]

    runner_pos = Vec(e_x, e_y)
    blocker_pos = Vec(b_x, b_y)

    blocker_distance_to_runner = blocker_pos.distance(runner_pos)
    blocker_angle_to_runner = blocker_pos.angle_diff(runner_pos)

    # Normalize blocker angle to runner
    blocker_angle = math.radians(b_angle) - blocker_angle_to_runner  # Normalize angle to runner
    blocker_angle_dot = vec_by_angle(blocker_angle).dot(Vec(1, 0))  # Cosine of blocker angle
    blocker_angle_cross = vec_by_angle(blocker_angle).cross(Vec(1, 0))  # Sine of blocker angle

    angle_v = Vec(0, 0).angle_diff(b_speed) - blocker_angle_to_runner  # Angle of velocity vector towards runner
    norm_v = math.sqrt(b_vx**2 + b_vy**2)
    blocker_v_dot = int(norm_v * vec_by_angle(angle_v).dot(Vec(1, 0)))  # Cosine of blocker speed angle
    blocker_v_cross = int(norm_v * vec_by_angle(angle_v).cross(Vec(1, 0)))  # Sine of blocker speed angle

    runner_angle_to_cp1 = runner_pos.angle_diff(CP1)

    # Normalize runner angle to CP1
    runner_angle = math.radians(e_angle) - runner_angle_to_cp1  # Normalize angle to CP1
    runner_angle_dot = vec_by_angle(runner_angle).dot(Vec(1, 0))  # Cosine of runner angle
    runner_angle_cross = vec_by_angle(runner_angle).cross(Vec(1, 0))  # Sine of runner angle

    angle_v = Vec(0, 0).angle_diff(e_speed) - runner_angle_to_cp1  # Angle of velocity vector towards CP1
    norm_v = math.sqrt(e_vx**2 + e_vy**2)
    runner_v_dot = int(norm_v * vec_by_angle(angle_v).dot(Vec(1, 0)))  # Cosine of runner speed angle
    runner_v_cross = int(norm_v * vec_by_angle(angle_v).cross(Vec(1, 0)))  # Sine of runner speed angle

    runner_distance_to_CP1 = runner_pos.distance(CP1)

    blocker_distance_to_CP1 = blocker_pos.distance(CP11)
    blocker_distance_to_CP2 = blocker_pos.distance(CP22)
    blocker_distance_to_CP3 = blocker_pos.distance(CP33)

    blocker_angle_to_CP1 = blocker_pos.angle_diff(CP11)
    blocker_angle_to_CP2 = blocker_pos.angle_diff(CP22)
    blocker_angle_to_CP3 = blocker_pos.angle_diff(CP33)

    blocker_angle_to_CP1 = math.radians(b_angle) - blocker_angle_to_CP1  # Normalize angle to CP1
    blocker_angle_to_CP2 = math.radians(b_angle) - blocker_angle_to_CP2  # Normalize angle to CP2
    blocker_angle_to_CP3 = math.radians(b_angle) - blocker_angle_to_CP3  # Normalize angle to CP3

    blocker_angle_dot_CP1 = vec_by_angle(blocker_angle_to_CP1).dot(Vec(1, 0))  # Cosine of blocker angle to CP1
    blocker_angle_cross_CP1 = vec_by_angle(blocker_angle_to_CP1).cross(Vec(1, 0))  # Sine of blocker angle to CP1
    blocker_angle_dot_CP2 = vec_by_angle(blocker_angle_to_CP2).dot(Vec(1, 0))  # Cosine of blocker angle to CP2
    blocker_angle_cross_CP2 = vec_by_angle(blocker_angle_to_CP2).cross(Vec(1, 0))  # Sine of blocker angle to CP2
    blocker_angle_dot_CP3 = vec_by_angle(blocker_angle_to_CP3).dot(Vec(1, 0))  # Cosine of blocker angle to CP3
    blocker_angle_cross_CP3 = vec_by_angle(blocker_angle_to_CP3).cross(Vec(1, 0))  # Sine of blocker angle to CP3

    return np.array(
        [
            blocker_distance_to_runner / 20000.0,  # Normalize distance to runner
            blocker_angle_dot,
            blocker_angle_cross,
            blocker_v_dot / 1000.0,  # Normalize blocker speed components
            blocker_v_cross / 1000.0,  # Normalize blocker speed components
            runner_distance_to_CP1 / 20000.0,  # Normalize distance to CP1
            runner_angle_dot,
            runner_angle_cross,
            runner_v_dot / 1000.0,  # Normalize runner speed components
            runner_v_cross / 1000.0,  # Normalize runner speed components
            blocker_distance_to_CP1 / 20000.0,  # Normalize distance to CP1
            blocker_angle_dot_CP1,
            blocker_angle_cross_CP1,
            blocker_distance_to_CP2 / 20000.0,  # Normalize distance to CP2
            blocker_angle_dot_CP2,
            blocker_angle_cross_CP2,
            blocker_distance_to_CP3 / 20000.0,  # Normalize distance to CP3
            blocker_angle_dot_CP3,
            blocker_angle_cross_CP3,
        ]
    )


# Comprehensive action space using for loops
ACTIONS = []

# Generate actions systematically using for loops
angle_deltas = [-18, 0, 18]
thrust_values = [0, 200]


for thrust in thrust_values:
    for angle_delta in angle_deltas:
        ACTIONS.append((angle_delta, thrust))

ACTIONS.append((0, "SHIELD"))  # Add SHIELD action


def get_output_runner(x, y, vx, vy, angle, next_cp_id, data):
    outs = runner_agent.forward(data)

    action_index = np.argmax(outs)
    a, s = ACTIONS[action_index]

    if s == "SHIELD":
        return (x, y, "SHIELD")

    a = (angle + a) * np.pi / 180
    nx = int(x + math.cos(a) * 1000)
    ny = int(y + math.sin(a) * 1000)

    power = min(max(s, 0), 200)

    if s > 200:
        power = "BOOST"

    return (nx, ny, power)


def get_output_blocker(blocker, data):

    x, y, vx, vy, angle, next_cp_id = blocker

    outs = blocker_agent.forward(data)

    action_index = np.argmax(outs)
    a, s = ACTIONS[action_index]

    if s == "SHIELD":
        return (x, y, "SHIELD")

    a = (angle + a) * np.pi / 180
    nx = int(x + math.cos(a) * 1000)
    ny = int(y + math.sin(a) * 1000)

    power = min(max(s, 0), 200)

    if s > 200:
        power = "BOOST"

    return (nx, ny, power)


first_turn = True
prev_enemy_checkpoints = [-1, -1]
enemy_progress = [0, 0]
enemy_leader = 0

while True:
    my_inputs = []
    enemy_inputs = []

    for i in range(2):
        x, y, vx, vy, angle, next_check_point_id = [int(j) for j in input().split()]
        my_inputs.append((x, y, vx, vy, angle, next_check_point_id))
    for i in range(2):
        x, y, vx, vy, angle, next_check_point_id = [int(j) for j in input().split()]
        enemy_inputs.append((x, y, vx, vy, angle, next_check_point_id))

    runner = my_inputs[0]
    blocker = my_inputs[1]

    r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id = runner

    if first_turn:
        cp1 = checkpoints[r_next_checkpoint_id]
        thrust = "BOOST"
        print(cp1.x, cp1.y, thrust, "RUNNER")
        print(cp1.x, cp1.y, "100", "BLOCKER")
        first_turn = False
        continue

    # Update enemy leader
    for i in range(2):
        if prev_enemy_checkpoints[i] != enemy_inputs[i][5]:
            enemy_progress[i] += 1

        if enemy_progress[i] > enemy_progress[1 - i]:
            enemy_leader = i

        prev_enemy_checkpoints[i] = enemy_inputs[i][5]

    # Use enemy blocker info
    enemy_blocker_index = 1 - enemy_leader
    enemy_blocker = enemy_inputs[enemy_blocker_index]
    b_x, b_y, b_vx, b_vy, b_angle, _ = enemy_blocker

    # Runner logic
    data = get_input_runner(
        r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id, b_x, b_y, b_vx, b_vy
    )  # For 13 input runner

    out = get_output_runner(r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id, data)
    print(*out, "RUNNER")

    #### Blocker logic
    enemy_to_block = enemy_inputs[enemy_leader]
    data = get_input_blocker(blocker, enemy_to_block)
    out = get_output_blocker(blocker, data)
    print(*out, "BLOCKER")
