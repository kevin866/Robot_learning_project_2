import mediapipe as mp
import numpy as np
import cv2
from collections import deque



def check_hand_closing(finger_history,hand_landmarks,palm_center):
    moving_fingers = 0
    for finger_tip in finger_history.keys():
        tip = np.array([hand_landmarks.landmark[finger_tip].x, hand_landmarks.landmark[finger_tip].y])

        # Compute RELATIVE distance to wrist
        relative_distance = np.linalg.norm(tip - palm_center)

        # Store distance history
        finger_history[finger_tip].append(relative_distance)

        # Compute change in relative distance
        if len(finger_history[finger_tip]) >= 2:
            delta = finger_history[finger_tip][-1] - finger_history[finger_tip][0]

            # If finger is closing
            if delta < -movement_threshold:
                moving_fingers += 1
            elif delta > movement_threshold:
                moving_fingers -= 1  # Finger extending
    return moving_fingers



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Store history of relative distances (finger to wrist)
history_length = 5  # Number of frames to track changes
finger_history = {8: deque(maxlen=history_length),  # Index finger tip
                  12: deque(maxlen=history_length), # Middle finger tip
                  16: deque(maxlen=history_length), # Ring finger tip
                  20: deque(maxlen=history_length)} # Pinky finger tip
# Store wrist position history
wrist_history = deque(maxlen=5)  # Store last 5 wrist positions
hand_events = {"close": [], "open": []}
# State tracking
hand_state = "opened"
movement_threshold = 0.025  # Adjust sensitivity
wrist_thre = 0.05
# cap = cv2.VideoCapture("your_video.mp4")  # Change to your video file
# cap = cv2.VideoCapture(0)  # Open webcam
cap = cv2.VideoCapture('v3.MOV')  # Open webcam

ignore_this_frame = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute palm center (average of key palm landmarks)
            palm_indices = [0, 1, 5, 9, 13, 17]  # Wrist and major palm landmarks
            palm_center = np.mean([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y] for i in palm_indices], axis=0)

            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])

            # Compute wrist movement speed
            if len(wrist_history) >= 2:
                wrist_movement = np.linalg.norm(wrist_history[-1] - wrist_history[0])
                print(wrist_movement)
                if wrist_movement > wrist_thre:
                    ignore_this_frame = True  # Large wrist movement detected
                else:
                    ignore_this_frame = False

            wrist_history.append(wrist)

            moving_fingers = check_hand_closing(finger_history,hand_landmarks,palm_center)
            if not ignore_this_frame:
                # Check if hand is opening or closing (ignore whole-hand movement)
                if moving_fingers >= 3 and hand_state != "holding":
                    hand_state = "holding"
                    hand_events["close"].append(tuple(palm_center))
                elif moving_fingers <= -3 and hand_state != "opened":
                    hand_state = "opened"
                    hand_events["open"].append(tuple(palm_center))

            # Display state
            cv2.putText(frame, f"State: {hand_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(hand_events)

