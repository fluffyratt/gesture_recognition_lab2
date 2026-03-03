import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

standard_pinky_threshold = 40.0          # наскільки “зігнутий” мізинець
standard_ring_threshold = 40.0           # наскільки “зігнутий” безіменний
standard_index_middle_separation = 55.0  # мін. відстань між кінчиками 8 та 12 (щоб не “з'єднані”)
standard_hand_size = 240.0
standard_thumb_cover = 55.0              # наскільки великий палець “прикриває” кінчики 16/20


def get_points(hand_landmarks, w, h):
    return [(lm.x * w, abs(lm.y * h - h)) for lm in hand_landmarks.landmark]


def distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def get_hand_size(points):
    wrist = points[0]
    middle_tip = points[12]
    return distance(middle_tip, wrist)


def calculate_thresholds(points):
    new_hand_size = get_hand_size(points)
    ratio = new_hand_size / standard_hand_size
    return (
        standard_pinky_threshold * ratio,
        standard_ring_threshold * ratio,
        standard_index_middle_separation * ratio,
        standard_thumb_cover * ratio,
    )


def is_finger_straight_up(points, tip_id, pip_id, mcp_id):
    """ “палець вгору” означає: tip_y > pip_y > mcp_y """
    tip = points[tip_id]
    pip = points[pip_id]
    mcp = points[mcp_id]
    return tip[1] > pip[1] > mcp[1]


def is_finger_curled(points, tip_id, mcp_id, threshold):
    """Наближена перевірка “зігнутості”: кінчик близько до основи."""
    return distance(points[tip_id], points[mcp_id]) <= threshold


def is_palm_edge(points):
    """“Долоня ребром” -> у проєкції ширина долоні (5..17) стає малою
    відносно висоти руки (0..12).
    """
    hand_size = get_hand_size(points)
    if hand_size <= 1e-6:
        return False
    palm_width = distance(points[5], points[17])  # index_mcp -> pinky_mcp
    return (palm_width / hand_size) < 0.55


def is_letter_k(points, handedness):
    """Літера К (укр. дактиль):
    - долоня ребром до співрозмовника
    - вказівний (8) і середній (12) підняті вгору, НЕ з’єднані
    - безіменний (16) і мізинець (20) напівзігнуті
    - великий палець (4) прикриває нігтьові фаланги безіменного/мізинця
    """

    # tips
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]
    thumb_tip = points[4]

    pinky_threshold, ring_threshold, sep_threshold, thumb_cover_threshold = calculate_thresholds(points)

    palm_edge = is_palm_edge(points)

    index_straight = is_finger_straight_up(points, 8, 6, 5)
    middle_straight = is_finger_straight_up(points, 12, 10, 9)

    # “не з’єднані”
    separated = distance(index_tip, middle_tip) >= sep_threshold

    # “напівзігнуті”
    ring_curled = is_finger_curled(points, 16, 13, ring_threshold)
    pinky_curled = is_finger_curled(points, 20, 17, pinky_threshold)

    # великий палець прикриває кінчики 16/20 (наближено)
    thumb_near = (
        distance(thumb_tip, ring_tip) <= thumb_cover_threshold
        or distance(thumb_tip, pinky_tip) <= thumb_cover_threshold
    )

    min_x = min(ring_tip[0], pinky_tip[0])
    max_x = max(ring_tip[0], pinky_tip[0])
    min_y = min(ring_tip[1], pinky_tip[1])
    max_y = max(ring_tip[1], pinky_tip[1])
    thumb_in_box = (
        (min_x - thumb_cover_threshold) <= thumb_tip[0] <= (max_x + thumb_cover_threshold)
        and (min_y - thumb_cover_threshold) <= thumb_tip[1] <= (max_y + thumb_cover_threshold)
    )
    thumb_covers = thumb_near or thumb_in_box

    return palm_edge and index_straight and middle_straight and separated and ring_curled and pinky_curled and thumb_covers


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Empty camera frame! 😨")
        continue

    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            h, w, _ = image.shape
            points = get_points(hand_landmarks, w, h)

            if is_letter_k(points, handedness):
                y_pos = 50 if handedness == "Left" else h - 50
                cv2.putText(image, f"K (ukr) recognised ({handedness} Hand)", (50, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
