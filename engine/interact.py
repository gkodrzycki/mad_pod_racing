# interact.py
import copy
import math
import sys
from typing import List, Optional, Tuple

import pygame

from engine.game_rule import GameHistory, GameSpec
from engine.game_sim import move_pods
from engine.util import check_point_radius, game_world_size, pod_force_field_radius
from engine.vec2 import Vec2, scalar_mul

# -----------------------------------------------------------------------------
#  Konfiguracja okna
# -----------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 1600, 900
FPS = 60

# identyczna skala jak w Gloss: 16000 → 1600 px  (9000 → 900 px)
SCALE = SCREEN_W / game_world_size.x
SHIFT_WORLD = scalar_mul(-0.5, game_world_size)  # przesunięcie świata
CENTER = (SCREEN_W // 2, SCREEN_H // 2)  # środek okna

# -----------------------------------------------------------------------------
#  Kolory
# -----------------------------------------------------------------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DIM_ORANGE = (128, 83, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
CYAN = (0, 255, 255)
PURPLE = (128, 0, 128)


# -----------------------------------------------------------------------------
#  Narzędzia rysunkowe
# -----------------------------------------------------------------------------
def world_to_screen(v: Vec2) -> Tuple[int, int]:
    """
    Rzutowanie współrzędnych świata na ekran:
      1. świat przesunięty o -½*world_size (tak jak w Gloss)
      2. przeskalowany
      3. odłożony względem środka okna
      4. oś Y odwrócona (pygame ma Y↓, Gloss Y↑)
    """
    wx, wy = (v + SHIFT_WORLD).x * SCALE, (v + SHIFT_WORLD).y * SCALE
    sx = int(CENTER[0] + wx)
    sy = int(CENTER[1] + wy)
    return sx, sy


def triangle(center: Tuple[int, int], size: int, angle: float):
    """Równoramienny trójkąt skierowany w prawo, obrócony o -angle (rad)."""
    cx, cy = center
    pts = [(size, 0), (-size * 0.5, size * 0.5), (-size * 0.5, -size * 0.5)]
    ca, sa = math.cos(angle), math.sin(angle)
    return [(int(cx + px * ca - py * sa), int(cy + px * sa + py * ca)) for px, py in pts]


# -----------------------------------------------------------------------------
#  Funkcje rysujące pojedyncze elementy
# -----------------------------------------------------------------------------
def draw_checkpoint(screen: pygame.Surface, pos: Vec2, idx: int):
    x, y = world_to_screen(pos)
    r = round(check_point_radius * SCALE)
    pygame.draw.circle(screen, GREEN, (x, y), r, 2)
    txt = FONT_S.render(str(idx), True, WHITE)
    screen.blit(txt, txt.get_rect(center=(x, y)))


def draw_pod(screen: pygame.Surface, pod, color):
    pos = world_to_screen(pod.pod_position)
    tgt = world_to_screen(pod.pod_movement.target)
    pygame.draw.line(screen, color, pos, tgt, 2)

    rad = round(pod_force_field_radius * SCALE)

    # Check if pod is using shield or boost
    from engine.game_sim import Boost, Shield

    is_shield = isinstance(pod.pod_movement.thrust, Shield)
    is_boost = isinstance(pod.pod_movement.thrust, Boost)

    if is_shield:
        # Shield animation - pulsing cyan circle
        import time

        pulse = abs(math.sin(time.time() * 8)) * 0.5 + 0.5  # 0.5 to 1.0
        shield_color = (
            int(CYAN[0] * pulse),
            int(CYAN[1] * pulse),
            int(CYAN[2] * pulse),
        )
        pygame.draw.circle(screen, shield_color, pos, rad + 5, 3)
        pygame.draw.circle(screen, CYAN, pos, rad, 2)
    elif is_boost:
        # Boost animation - pulsing orange with flame trails
        import time

        pulse = abs(math.sin(time.time() * 12)) * 0.4 + 0.6  # 0.6 to 1.0
        boost_color = (int(ORANGE[0] * pulse), int(ORANGE[1] * pulse), 0)

        # Draw multiple expanding rings for boost effect
        for i in range(3):
            ring_offset = i * 3
            ring_alpha = 1.0 - (i * 0.3)
            ring_color = (
                int(ORANGE[0] * pulse * ring_alpha),
                int(ORANGE[1] * pulse * ring_alpha),
                0,
            )
            pygame.draw.circle(screen, ring_color, pos, rad + ring_offset, 2)

        # Draw main pod circle
        pygame.draw.circle(screen, boost_color, pos, rad, 3)
    else:
        pygame.draw.circle(screen, color, pos, rad)

    pygame.draw.polygon(screen, YELLOW, triangle(pos, rad, pod.pod_angle or 0.0))


def draw_legend(screen, idx, color, name, is_winner):
    y = 20 + idx * 40
    pygame.draw.circle(screen, color, (30, y), 12)
    txt = FONT_M.render(name, True, color)
    screen.blit(txt, (50, y - txt.get_height() // 2))

    tag = "<< 1st >>" if is_winner else ""
    tag_col = ORANGE if is_winner else DIM_ORANGE
    tag_txt = FONT_M.render(tag, True, tag_col)
    screen.blit(tag_txt, (200, y - tag_txt.get_height() // 2))


def draw_action_info(screen: pygame.Surface, pods, frame_idx: int):
    """Draw rectangles with current action information for each pod."""
    from engine.game_sim import Boost, Normal, Shield

    rect_width, rect_height = 220, 90
    margin = 10

    for i, pod in enumerate(pods):
        # Position rectangles on the right side of screen
        x = SCREEN_W - rect_width - margin
        y = margin + i * (rect_height + margin)

        # Draw background rectangle - pod 0 is P1, pod 1 is P2
        color = BLUE if i == 0 else RED
        pygame.draw.rect(screen, DARK_GRAY, (x, y, rect_width, rect_height))
        pygame.draw.rect(screen, color, (x, y, rect_width, rect_height), 2)

        # Pod identification - now 1 pod per player
        pod_name = f"Pod {i+1} ({'P1' if i == 0 else 'P2'})"
        name_txt = FONT_S.render(pod_name, True, WHITE)
        screen.blit(name_txt, (x + 5, y + 5))

        # Action information
        if hasattr(pod, "pod_movement") and pod.pod_movement:
            target = pod.pod_movement.target
            thrust_obj = pod.pod_movement.thrust

            position_txt = f"Position: ({int(pod.pod_position.x)}, {int(pod.pod_position.y)})"
            position_render = FONT_S.render(position_txt, True, WHITE)
            screen.blit(position_render, (x + 5, y + 65))

            # Target coordinates
            target_txt = f"Target: ({int(target.x)}, {int(target.y)})"
            target_render = FONT_S.render(target_txt, True, WHITE)
            screen.blit(target_render, (x + 5, y + 20))

            # Thrust value or action type
            if isinstance(thrust_obj, Shield):
                thrust_txt = "Action: SHIELD"
                thrust_color = CYAN
            elif isinstance(thrust_obj, Boost):
                thrust_txt = "Action: BOOST"
                thrust_color = ORANGE
            elif isinstance(thrust_obj, Normal):
                thrust_txt = f"Thrust: {thrust_obj.n}"
                thrust_color = WHITE
            else:
                thrust_txt = f"Thrust: {str(thrust_obj)}"
                thrust_color = WHITE

            thrust_render = FONT_S.render(thrust_txt, True, thrust_color)
            screen.blit(thrust_render, (x + 5, y + 35))

            # Speed information
            if hasattr(pod, "pod_speed"):
                speed = math.sqrt(pod.pod_speed.x**2 + pod.pod_speed.y**2)
                speed_txt = f"Speed: {int(speed)}"
                speed_render = FONT_S.render(speed_txt, True, WHITE)
                screen.blit(speed_render, (x + 5, y + 50))
        else:
            no_action_txt = FONT_S.render("No action data", True, GRAY)
            screen.blit(no_action_txt, (x + 5, y + 35))


def draw_frame_slider(
    screen: pygame.Surface,
    current_frame: int,
    total_frames: int,
    mouse_pos: Tuple[int, int],
    mouse_clicked: bool,
) -> Optional[int]:
    """Draw a frame slider and return new frame if clicked"""
    slider_x = 50
    slider_y = SCREEN_H - 30
    slider_width = SCREEN_W - 100
    slider_height = 10

    # Draw slider background
    pygame.draw.rect(screen, DARK_GRAY, (slider_x, slider_y, slider_width, slider_height))
    pygame.draw.rect(screen, WHITE, (slider_x, slider_y, slider_width, slider_height), 1)

    # Calculate slider position
    if total_frames > 1:
        slider_pos = slider_x + (current_frame / (total_frames - 1)) * slider_width
    else:
        slider_pos = slider_x

    # Draw slider handle
    handle_size = 8
    pygame.draw.circle(screen, ORANGE, (int(slider_pos), slider_y + slider_height // 2), handle_size)

    # Draw frame numbers at ends
    start_txt = FONT_S.render("1", True, WHITE)
    end_txt = FONT_S.render(str(total_frames), True, WHITE)
    screen.blit(start_txt, (slider_x - 15, slider_y - 5))
    screen.blit(end_txt, (slider_x + slider_width + 5, slider_y - 5))

    # Check for mouse interaction
    if mouse_clicked:
        mx, my = mouse_pos
        if slider_y - handle_size <= my <= slider_y + slider_height + handle_size:
            if slider_x <= mx <= slider_x + slider_width:
                # Calculate new frame based on mouse position
                relative_pos = (mx - slider_x) / slider_width
                new_frame = int(relative_pos * (total_frames - 1))
                return max(0, min(new_frame, total_frames - 1))

    return None


def draw_frame_info(
    screen: pygame.Surface,
    frame_idx: int,
    total_frames: int,
    is_paused: bool,
    show_ui: bool,
):
    """Draw frame counter rectangle with pause status."""
    if not show_ui:
        return

    rect_width, rect_height = 180, 90
    x = SCREEN_W - rect_width - 10
    y = SCREEN_H - rect_height - 40  # Move up to make room for slider

    pygame.draw.rect(screen, DARK_GRAY, (x, y, rect_width, rect_height))
    pygame.draw.rect(screen, WHITE, (x, y, rect_width, rect_height), 1)

    frame_txt = f"Frame: {frame_idx + 1}/{total_frames}"
    frame_render = FONT_S.render(frame_txt, True, WHITE)
    screen.blit(frame_render, (x + 5, y + 5))

    # Pause status
    status_txt = "PAUSED" if is_paused else "PLAYING"
    status_color = ORANGE if is_paused else GREEN
    status_render = FONT_S.render(status_txt, True, status_color)
    screen.blit(status_render, (x + 5, y + 20))

    # Controls info
    controls_txt = "SPACE: Pause | ←→: Skip"
    controls_render = FONT_S.render(controls_txt, True, GRAY)
    screen.blit(controls_render, (x + 5, y + 35))

    # UI toggle info
    ui_txt = "H: Hide UI"
    ui_render = FONT_S.render(ui_txt, True, GRAY)
    screen.blit(ui_render, (x + 5, y + 50))

    # Slider info
    slider_txt = "Click slider to jump"
    slider_render = FONT_S.render(slider_txt, True, GRAY)
    screen.blit(slider_render, (x + 5, y + 65))


# -----------------------------------------------------------------------------
#  Główna animacja
# -----------------------------------------------------------------------------
def game_animate(
    names: List[str],
    turn_per_sec: float,
    spec: GameSpec,
    history: GameHistory,
) -> None:
    pygame.init()
    global FONT_S, FONT_M
    FONT_S = pygame.font.SysFont(None, 18)
    FONT_M = pygame.font.SysFont(None, 24)

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CSB replay (pygame)")
    clock = pygame.time.Clock()

    frames = history
    n_frames = len(frames)

    idx, frac_t = 0, 0.0
    is_paused = False
    show_ui = True
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        if not is_paused:
            frac_t += dt * turn_per_sec

        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False

        # zdarzenia
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:  # Left click
                    mouse_clicked = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RIGHT:
                    idx, frac_t = min(idx + 1, n_frames - 1), 0.0
                elif e.key == pygame.K_LEFT:
                    idx, frac_t = max(idx - 1, 0), 0.0
                elif e.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif e.key == pygame.K_h:
                    show_ui = not show_ui
                elif e.key == pygame.K_ESCAPE:
                    running = False

        # Check slider interaction
        if show_ui:
            new_frame = draw_frame_slider(screen, idx, n_frames, mouse_pos, mouse_clicked)
            if new_frame is not None:
                idx, frac_t = new_frame, 0.0
                is_paused = True  # Pause when user manually seeks

        # automatyczne przechodzenie do kolejnej klatki (tylko gdy nie jest pauzowane)
        while not is_paused and frac_t >= 1.0 and idx < n_frames - 1:
            frac_t -= 1.0
            idx += 1

        # rysowanie
        screen.fill(BLACK)
        for i, ck in enumerate(spec.checkpoints):
            draw_checkpoint(screen, ck, i)

        # interpolacja pomiędzy klatkami – głęboka kopia zamiast .copy()
        base = frames[idx]
        pods_now = move_pods(frac_t, copy.deepcopy(base)) if (frac_t and idx < n_frames - 1) else base

        for pi, pod in enumerate(pods_now):
            draw_pod(screen, pod, BLUE if pi == 0 else RED)

        # Draw UI elements only if show_ui is True
        if show_ui:
            draw_action_info(screen, pods_now, idx)
            draw_frame_info(screen, idx, n_frames, is_paused, show_ui)

            if len(names) == 2:
                p1_name, p2_name = names
            else:
                p1_name = names[0]
            # Use current frame's checkpoint status for real-time winner display
            ck_left = [len(p.pod_next_checkpoints) for p in pods_now]
            # Handle both 2-pod and 4-pod games
            if len(ck_left) >= 4:
                p1_win = min(ck_left[0], ck_left[1]) < min(ck_left[2], ck_left[3])
            elif len(ck_left) == 2:
                p1_win = ck_left[0] < ck_left[1]
            else:
                p1_win = True

            draw_legend(screen, 0, BLUE, p1_name, p1_win)
            if len(names) == 2:
                draw_legend(screen, 1, RED, p2_name, not p1_win)

            # Always draw slider last so it's on top
            draw_frame_slider(screen, idx, n_frames, mouse_pos, False)
        else:
            draw_frame_info(screen, idx, n_frames, is_paused, show_ui)

        pygame.display.flip()

        if idx == n_frames - 1 and frac_t == 0.0 and not is_paused:
            running = False

    pygame.quit()
    sys.exit()


# -----------------------------------------------------------------------------
#  Szybki helper
# -----------------------------------------------------------------------------
def winner(history: GameHistory) -> int:
    ck_left = [len(p.pod_next_checkpoints) for p in history[-1]]  # Use final state
    # Handle both 2-pod and 4-pod games
    if len(ck_left) >= 4:
        return 1 if min(ck_left[0], ck_left[1]) < min(ck_left[2], ck_left[3]) else 2
    elif len(ck_left) == 2:
        return 1 if ck_left[0] < ck_left[1] else 2
    else:
        return 1 if ck_left[0] == 0 else 2
