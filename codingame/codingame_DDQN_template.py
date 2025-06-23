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
    (32, runner_input),  # fc1 weights
    (32,),  # fc1 biases
    (32, 32),  # fc2 weights
    (32,),  # fc2 biases
    (32, 32),  # fc3 weights
    (32,),  # fc3 biases
    (32, 32),  # fc4 weights
    (32,),  # fc4 biases
    (32, 32),  # fc5 weights
    (32,),  # fc5 biases
    (runner_output, 32),  # out weights
    (runner_output,),  # out biases
]

counts = [np.prod(shape) for shape in layer_shapes]

offset = 0
decoded_layers = []
for cnt, shape in zip(counts, layer_shapes):
    chunk = all_weights_runner[offset : offset + cnt]
    decoded = decode_unicode_weights(chunk, shape, scale_factor=scale_factor, offset=offset_unicode)
    decoded_layers.append(decoded)
    offset += cnt

w0c, b0c, w1c, b1c, w2c, b2c, w3c, b3c, w4c, b4c, w5c, b5c = decoded_layers


class SimpleQNetworkRunner:
    def relu(self, x):
        return [max(0, val) for val in x]

    def tanh(self, x):
        return [np.tanh(val) for val in x]

    def __init__(self):
        self.fc1_weight = w0c
        self.fc1_bias = b0c
        self.fc2_weight = w1c
        self.fc2_bias = b1c
        self.fc3_weight = w2c
        self.fc3_bias = b2c
        self.fc4_weight = w3c
        self.fc4_bias = b3c
        self.out_weight = w4c
        self.out_bias = b4c

        self.activation = self.tanh

    def forward(self, x):
        x = np.array(x)
        x = self.activation(self.fc1_weight @ x + self.fc1_bias)
        x = self.activation(self.fc2_weight @ x + self.fc2_bias)
        x = self.activation(self.fc3_weight @ x + self.fc3_bias)
        x = self.activation(self.fc4_weight @ x + self.fc4_bias)
        return self.out_weight @ x + self.out_bias


layer_shapes = [
    (32, blocker_input),  # fc1 weights
    (32,),  # fc1 biases
    (32, 32),  # fc2 weights
    (32,),  # fc2 biases
    (32, 32),  # fc3 weights
    (32,),  # fc3 biases
    (32, 32),  # fc4 weights
    (32,),  # fc4 biases
    (32, 32),  # fc5 weights
    (32,),  # fc5 biases
    (blocker_output, 32),  # out weights
    (blocker_output,),  # out biases
]

counts = [np.prod(shape) for shape in layer_shapes]

offset = 0
decoded_layers = []
for cnt, shape in zip(counts, layer_shapes):
    chunk = all_weights_blocker[offset : offset + cnt]
    decoded = decode_unicode_weights(chunk, shape, scale_factor=scale_factor, offset=offset_unicode)
    decoded_layers.append(decoded)
    offset += cnt

w0c_b, b0c_b, w1c_b, b1c_b, w2c_b, b2c_b, w3c_b, b3c_b, w4c_b, b4c_b, w5c_b, b5c_b = decoded_layers


class SimpleQNetworkBlocker:
    def relu(self, x):
        return [max(0, val) for val in x]

    def tanh(self, x):
        return [np.tanh(val) for val in x]

    def __init__(self):
        self.fc1_weight = w0c_b
        self.fc1_bias = b0c_b
        self.fc2_weight = w1c_b
        self.fc2_bias = b1c_b
        self.fc3_weight = w2c_b
        self.fc3_bias = b2c_b
        self.fc4_weight = w3c_b
        self.fc4_bias = b3c_b
        self.out_weight = w4c_b
        self.out_bias = b4c_b

        self.activation = self.tanh

    def forward(self, x):
        x = np.array(x)
        x = self.activation(self.fc1_weight @ x + self.fc1_bias)
        x = self.activation(self.fc2_weight @ x + self.fc2_bias)
        x = self.activation(self.fc3_weight @ x + self.fc3_bias)
        x = self.activation(self.fc4_weight @ x + self.fc4_bias)
        return self.out_weight @ x + self.out_bias


runner_agent = SimpleQNetworkRunner()
blocker_agent = SimpleQNetworkBlocker()


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


def get_input_runner(x, y, vx, vy, angle, next_cp_id):
    cp1 = checkpoints[next_cp_id]
    cp2 = checkpoints[(next_cp_id + 1) % checkpoint_count]

    angle *= np.pi / 180

    pod = Vec(x, y)
    speed = Vec(vx, vy)

    distance = pod.distance(cp1)
    angle_to_cp1 = pod.angle_diff(cp1)

    angle -= angle_to_cp1

    angle_v = Vec(0, 0).angle_diff(speed) - angle_to_cp1

    norm_v = math.sqrt(vx**2 + vy**2)

    angle_next_dir = cp1.angle_diff(cp2) - angle_to_cp1
    distance_cp1_cp2 = cp1.distance(cp2)

    angle_dot = vec_by_angle(angle).dot(Vec(1, 0))
    angle_cross = vec_by_angle(angle).cross(Vec(1, 0))

    v_dot = int(norm_v * vec_by_angle(angle_v).dot(Vec(1, 0)))
    v_cross = int(norm_v * vec_by_angle(angle_v).cross(Vec(1, 0)))

    next_dir_dot = vec_by_angle(angle_next_dir).dot(Vec(1, 0))
    next_dir_cross = vec_by_angle(angle_next_dir).cross(Vec(1, 0))

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

    r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id = runner

    if first_turn:
        cp1 = checkpoints[r_next_checkpoint_id]
        thrust = "BOOST"
        print(cp1.x, cp1.y, thrust, "RUNNER")
        print(cp1.x, cp1.y, "100", "BLOCKER")
        first_turn = False
        continue

    data = get_input_runner(r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id)
    out = get_output_runner(r_x, r_y, r_vx, r_vy, r_angle, r_next_checkpoint_id, data)
    print(*out, "RUNNER")

    for i in range(2):
        if prev_enemy_checkpoints[i] != enemy_inputs[i][5]:
            enemy_progress[i] += 1

        if enemy_progress[i] > enemy_progress[1 - i]:
            enemy_leader = i

        prev_enemy_checkpoints[i] = enemy_inputs[i][5]

    blocker = my_inputs[1]
    enemy_to_block = enemy_inputs[enemy_leader]

    data = get_input_blocker(blocker, enemy_to_block)
    out = get_output_blocker(blocker, data)
    print(*out, "BLOCKER")
