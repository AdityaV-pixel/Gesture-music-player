import os
import time
import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Initialize Pygame Mixer
pygame.mixer.init()

# Path to your swaras folder (including the sounds_swaras subfolder)
SWARA_DIR = os.path.join("Week 4 Swaras", "sounds_swaras")

# Exact filenames matching your folder structure
swara_files = {
    "Sa": os.path.join(SWARA_DIR, "1_Sa.mp3"),
    "Re": os.path.join(SWARA_DIR, "2_Re.mp3"),
    "Ga": os.path.join(SWARA_DIR, "3_Ga.mp3"),
    "Ma": os.path.join(SWARA_DIR, "4_Ma.mp3"),
    "Pa": os.path.join(SWARA_DIR, "5_Pa.mp3"),
    "Dha": os.path.join(SWARA_DIR, "6_Dha.mp3"),
    "Ni": os.path.join(SWARA_DIR, "7_Ni.mp3"),
    "Sa_upper": os.path.join(SWARA_DIR, "8_Sa_upper.mp3"),
}

# Load sounds into a dictionary
sounds = {}
for name, path in swara_files.items():
    try:
        sounds[name] = pygame.mixer.Sound(path)
        print(f"Successfully loaded swara: {name}")
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")

# Path to the model
MODEL_PATH = "hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,  # Support up to 2 hands for full 8-note range
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.time()

print("Starting Gesture Swara Player... Press 'q' to quit.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                  # Palm base connection
]

current_swara = None
cooldown = 0
swara_list = ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni", "Sa_upper"]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    detected_swara = None
    total_fingers_up = 0

    if result.hand_landmarks:
        # Loop through ALL detected hands (up to 2) and sum their fingers
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            
            # Draw skeleton for every hand detected
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
            for x, y in points:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            # Robust distance check from wrist (Point 0)
            wrist = points[0]
            fingers_up = 0
            
            # Finger pairs: Tip vs Knuckle (MCP)
            finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
            for tip_idx, mcp_idx in finger_pairs:
                dist_tip = ((points[tip_idx][0] - wrist[0])**2 + (points[tip_idx][1] - wrist[1])**2) ** 0.5
                dist_mcp = ((points[mcp_idx][0] - wrist[0])**2 + (points[mcp_idx][1] - wrist[1])**2) ** 0.5
                if dist_tip > dist_mcp * 1.15:
                    fingers_up += 1

            # Thumb check
            thumb_dist = ((points[4][0] - wrist[0])**2 + (points[4][1] - wrist[1])**2) ** 0.5
            thumb_base_dist = ((points[2][0] - wrist[0])**2 + (points[2][1] - wrist[1])**2) ** 0.5
            if thumb_dist > thumb_base_dist * 1.1:
                fingers_up += 1

            # Accumulate fingers from this hand
            total_fingers_up += fingers_up

        # Map total combined fingers across both hands (0 to 7+) to your 8 swaras
        index = min(max(0, total_fingers_up), len(swara_list) - 1)
        detected_swara = swara_list[index]

    if cooldown > 0:
        cooldown -= 1

    # Play audio if a new swara is detected and cooldown is clear
    if detected_swara and detected_swara != current_swara and cooldown == 0:
        pygame.mixer.stop()  # Stop previous sound
        if detected_swara in sounds:
            sounds[detected_swara].play()
            current_swara = detected_swara
            print(f"Playing Swara: {detected_swara} (Total fingers: {total_fingers_up})")
            cooldown = 20  # Frame cooldown

   # Display live finger count and current swara with clean spacing
    cv2.putText(frame, f"Fingers Detected: {total_fingers_up}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    swara_display = f"Swara: {current_swara}" if current_swara else "Swara: None"
    cv2.putText(frame, swara_display, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Gesture Swara Player", frame)
    cv2.imshow("Gesture Swara Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()