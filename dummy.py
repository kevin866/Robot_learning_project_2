import mediapipe as mp
import cv2
import math

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p0):
    """Calculates the angle between three points (p1, p2, p0) where p0 is the center."""
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p0
    angle1 = math.atan2(y1 - y0, x1 - x0)
    angle2 = math.atan2(y2 - y0, x2 - x0)
    return angle2 - angle1

def calculate_center(landmarks):
    """Estimate the palm center (a point between wrist, thumb, and index)."""
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_base = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_base = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Simple estimate of palm center (weighted average or midpoint)
    center_x = (wrist.x + thumb_base.x + index_base.x) / 3
    center_y = (wrist.y + thumb_base.y + index_base.y) / 3
    return (center_x, center_y)

# Setup camera
cap = cv2.VideoCapture('v7.MOV')

# Initialize previous positions and angles to None
previous_thumb_angle = None
previous_index_angle = None
central_point = None

# Angular velocity threshold for detecting pouring action
ANGULAR_VELOCITY_THRESHOLD = 0.1  # Adjust this value as needed

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Estimate the palm center (rotation center)
                central_point = calculate_center(landmarks)
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate the angle of the thumb and index finger relative to the palm center
                thumb_angle = calculate_angle((thumb_tip.x, thumb_tip.y), (central_point[0], central_point[1]), (thumb_tip.x, thumb_tip.y))
                index_angle = calculate_angle((index_tip.x, index_tip.y), (central_point[0], central_point[1]), (index_tip.x, index_tip.y))

                # Calculate the angular velocity (rate of change of the angle)
                thumb_angular_velocity = 0
                index_angular_velocity = 0

                if previous_thumb_angle is not None:
                    thumb_angular_velocity = (thumb_angle - previous_thumb_angle)
                if previous_index_angle is not None:
                    index_angular_velocity = (index_angle - previous_index_angle)

                # Check if the angular velocities are above a threshold indicating rotation
                if abs(thumb_angular_velocity) > ANGULAR_VELOCITY_THRESHOLD or abs(index_angular_velocity) > ANGULAR_VELOCITY_THRESHOLD:
                    print("Possible Rotation Detected (pouring action)")

                # Store previous angles for comparison
                previous_thumb_angle = thumb_angle
                previous_index_angle = index_angle

                # Display angular velocity for debugging
                cv2.putText(frame, f'Thumb Angular Vel: {thumb_angular_velocity:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f'Index Angular Vel: {index_angular_velocity:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with the drawn landmarks
        cv2.imshow('Hand Rotation Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
