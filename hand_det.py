import cv2
import mediapipe as mp
import numpy as np

def is_holding(hand_landmarks):
    # Extract key points
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
    ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])

    # Compute distances to wrist
    index_dist = np.linalg.norm(index_tip - wrist)
    middle_dist = np.linalg.norm(middle_tip - wrist)
    ring_dist = np.linalg.norm(ring_tip - wrist)
    pinky_dist = np.linalg.norm(pinky_tip - wrist)

    # Threshold for grasp detection (tune experimentally)
    threshold = 0.1  # Adjust based on hand size in image

    # If at least 3 fingers are curled in, classify as "holding"
    curled_fingers = sum(dist < threshold for dist in [index_dist, middle_dist, ring_dist, pinky_dist])
    print(middle_dist)
    
    return curled_fingers >= 3  # Return True if gripping


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open webcam
# video_path = "video.MOV"  # Change this to your video file path
# cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Simple action recognition example (fist, open hand)
            if is_holding(hand_landmarks):  # Hand closed (fist)
                action = "is holding"
            else:
                action = "Opening"

            # Display action
            cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
