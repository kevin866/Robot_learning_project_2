import mediapipe as mp
import numpy as np
import cv2
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize deque to store past finger distances (for smoothing)
history_length = 5  # Number of frames to track changes
finger_history = {8: deque(maxlen=history_length),  # Index finger tip
                  12: deque(maxlen=history_length), # Middle finger tip
                  16: deque(maxlen=history_length), # Ring finger tip
                  20: deque(maxlen=history_length)} # Pinky finger tip

# Threshold for detecting opening/closing
movement_threshold = 0.02  # Adjust based on hand size in image

# State variable for hand status
hand_state = "opened"  # Initial state

# Open video
# cap = cv2.VideoCapture("video.MOV")  # Change to your video file
cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])

            # Track distances for index, middle, ring, and pinky
            moving_fingers = 0
            for finger_tip in finger_history.keys():
                tip = np.array([hand_landmarks.landmark[finger_tip].x, hand_landmarks.landmark[finger_tip].y])
                distance = np.linalg.norm(tip - wrist)

                # Store distance history
                finger_history[finger_tip].append(distance)

                # Compute rate of change if enough history is stored
                if len(finger_history[finger_tip]) >= 2:
                    delta = finger_history[finger_tip][-1] - finger_history[finger_tip][0]

                    # Check if finger is closing
                    if delta < -movement_threshold:
                        moving_fingers += 1
                    elif delta > movement_threshold:
                        moving_fingers -= 1  # Count fingers moving away (opening)

            # Detect hand action
            if moving_fingers >= 3 and hand_state != "holding":
                hand_state = "holding"
            elif moving_fingers <= -3 and hand_state != "opened":
                hand_state = "opened"

            # Display the hand state
            cv2.putText(frame, f"State: {hand_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
