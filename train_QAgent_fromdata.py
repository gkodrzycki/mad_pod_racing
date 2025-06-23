import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from engine.model import QNetwork
from engine.model_dueling import DuelingMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(filename):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # pomiń nagłówek

    X, Y = [], []
    for i in range(0, len(lines), 2):
        inputs = list(map(float, lines[i].strip().split()))
        outputs = list(map(float, lines[i + 1].strip().split()))
        X.append(inputs)
        Y.append(outputs)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32, device=device)


# Wczytaj dane
X, Y = load_dataset("data.txt")  # <- zmień na nazwę swojego pliku


ACTIONS = []

# Generate actions systematically using for loops
angle_deltas = [-18, 0, 18]
thrust_values = [0, 100]

for thrust in thrust_values:
    for angle_delta in angle_deltas:
        ACTIONS.append((angle_delta, thrust))

# Konwersja Y na skwantyzowane indeksy
angle = (Y[:, 0] * 36 - 18).round()  # zaokrąglenie dla bezpieczeństwa
thrust = (Y[:, 1] * 200).round()

# Zakodowanie angle jako indeks (0, 1, 2)
# -18 → 0, 0 → 1, 18 → 2
angle_idx = ((angle + 18) // 18).long().clamp(0, 2)

# Zakodowanie thrust jako offset (0 dla 0, 3 dla 100)
thrust_offset = torch.where(thrust >= 50, 3, 0)

# Finalny indeks w ACTIONS
Y_ind = (angle_idx + thrust_offset).tolist()

Y_ind = torch.tensor(Y_ind, dtype=torch.long, device=device)

# Utwórz model
model = DuelingMLP(state_dim=8, action_dim=len(ACTIONS))  # wejście 8, wyjście 6
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Trenowanie
epochs = 1000
batch_size = 512
for epoch in tqdm(range(epochs)):
    perm = torch.randperm(X.size(0))
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    for i in range(0, X.size(0), batch_size):
        xb = X_shuffled[i : i + batch_size]
        yb = Y_shuffled[i : i + batch_size]

        optimizer.zero_grad()

        pred = model(xb).to(device)  # shape: (batch_size, num_actions)
        # target = F.one_hot(yb, num_classes=len(ACTIONS)).float()  # one-hot encoding

        loss = criterion(pred, yb)  # yb: class indices
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "models_weights/dueling_pretrained.pkl",
)
