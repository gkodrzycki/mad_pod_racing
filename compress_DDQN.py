import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from engine.model import QNetwork

# ------- Compression Utilities -------
SCALE = 512.0
OFFSET = 12.0


def _decode_from_unicode(unicode_str: str) -> np.ndarray:
    b = unicode_str.encode("utf-16-be")
    q = np.frombuffer(b, dtype=">i2").astype(np.int16)
    return q.astype(np.float32) / SCALE - OFFSET


def compress_weights_to_unicode(nn_weights: np.ndarray, epsilon: float = 1e-3) -> str:
    """
    Quantize and pack a 1D float32 array into a UTF‑16 string.
    Also verifies that dequantized weights deviate from originals by at most epsilon.
    """
    w = nn_weights.astype(np.float32).flatten()
    q = np.round((w + OFFSET) * SCALE).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    b = bytearray()
    for s in q:
        high = (int(s) >> 8) & 0xFF
        low = int(s) & 0xFF
        b.append(high)
        b.append(low)
    uni = b.decode("utf-16-be")

    w_rec = _decode_from_unicode(uni)
    diff = w_rec - w
    max_err = np.max(np.abs(diff))
    mean_err = np.mean(np.abs(diff))
    print(f"[compress] max_abs_error={max_err:.6e}, mean_abs_error={mean_err:.6e}")
    if max_err > epsilon:
        raise ValueError(f"Quantization error {max_err:.6e} exceeds epsilon={epsilon}")

    return uni


# ------- Extraction -------
def extract_all_weights(model: nn.Module) -> np.ndarray:
    params = []
    for _, p in model.named_parameters():
        params.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(params)


# ------- Template Copy & Replace -------
def write_codingame_bot(
    template_path: str,
    output_path: str,
    runner_weights_str: str,
    blocker_weights_str: str,
    runner_state_dim,
    runner_action_dim,
    blocker_state_dim,
    blocker_action_dim,
):
    shutil.copyfile(template_path, output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("<all_weights_runner>", runner_weights_str)
    content = content.replace("<all_weights_blocker>", blocker_weights_str)
    content = content.replace("<runner_input>", str(runner_state_dim))
    content = content.replace("<blocker_input>", str(blocker_state_dim))
    content = content.replace("<runner_output>", str(runner_action_dim))
    content = content.replace("<blocker_output>", str(blocker_action_dim))
    content = content.replace("<scale_factor>", str(SCALE))
    content = content.replace("<offset>", str(OFFSET))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------------- MAIN ------------------
runner_model_path = "absolute_path_to_your_runner_model"
blocker_model_path = "absolute_path_to_your_blocker_model"

output_file = "codingame/codingame_bot.py"

blocker_state_dim = 19
blocker_action_dim = 7

runner_state_dim = 8
runner_action_dim = 7


runner_model = QNetwork(state_dim=runner_state_dim, action_dim=runner_action_dim)
blocker_model = QNetwork(state_dim=blocker_state_dim, action_dim=blocker_action_dim)


checkpoint = torch.load(runner_model_path)
runner_model.load_state_dict(checkpoint["model_state_dict"])

checkpoint = torch.load(blocker_model_path)
blocker_model.load_state_dict(checkpoint["model_state_dict"])


# Compress and save weights
flat_r = extract_all_weights(runner_model)
uni_r = compress_weights_to_unicode(flat_r)

flat_b = extract_all_weights(blocker_model)
uni_b = compress_weights_to_unicode(flat_b)

write_codingame_bot(
    "codingame/codingame_DDQN_template.py",
    output_file,
    uni_r,
    uni_b,
    runner_state_dim,
    runner_action_dim,
    blocker_state_dim,
    blocker_action_dim,
)
print(f"Saved compressed weights ({len(uni_r) + len(uni_b)} chars) to {output_file}")
