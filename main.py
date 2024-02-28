import cv2
import mediapipe as mp
import numpy as np


def draw_on_canvas(canvas, point1, point2, color):
    cv2.line(canvas, point1, point2, color, 5)


def draw_cursor(frame, position, color, radius=10):
    cv2.circle(frame, position, radius, color, -1)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def detect_custom_gesture(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[8]
    index_finger = hand_landmarks.landmark[5]
    middle_finger_tip = hand_landmarks.landmark[12]


    if index_finger_tip.y < middle_finger_tip.y:
        distance_index_knuckle_tip = euclidean_distance((index_finger.x, index_finger.y), (index_finger_tip.x, index_finger_tip.y))
        if distance_index_knuckle_tip > 0.06:
            return True

    return False


def process_hand_landmarks(frame):
    global prev_point, prev_point_smooth, canvas, draw_color


    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        gesture_detected = False
        for hand_landmarks in results.multi_hand_landmarks:
            if detect_custom_gesture(hand_landmarks):
                gesture_detected = True
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                if prev_point is not None:
                    if prev_point_smooth is not None:
                        # Apply simple moving average filter
                        x = int((prev_point_smooth[0] + x) / 2)
                        y = int((prev_point_smooth[1] + y) / 2)
                    draw_on_canvas(canvas, prev_point, (x, y), draw_color)
                prev_point = (x, y)
                prev_point_smooth = (x, y)
                draw_cursor(frame, (x, y), (255, 0, 0))


        if not gesture_detected:
            prev_point = None
            prev_point_smooth = None


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

prev_point = None
prev_point_smooth = None
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
draw_color = (0, 0, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    process_hand_landmarks(frame)


    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)


    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        prev_point = None
        prev_point_smooth = None
    elif key == ord('r'):
        draw_color = (0, 0, 255)
    elif key == ord('b'):
        draw_color = (0, 0, 0)


cap.release()
cv2.destroyAllWindows()
