import cv2
import mediapipe as mp
import pyautogui

# Configurar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Configurar mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Configurar umbral de detección del dedo índice y estado anterior
INDEX_THRESHOLD = 0.8
prev_state = False

while True:
    success, image = cap.read()
    if not success:
        break

    # Convertir la imagen de BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar manos en la imagen
    results = hands.process(image)

    # Dibujar landmarks de las manos en la imagen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener la posición del centro de la mano
            center_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1]
            center_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]

            # Mover el mouse al centro de la mano
            pyautogui.moveTo(center_x, center_y)

            # Obtener la posición del dedo índice
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Verificar si se ha extendido el dedo índice
            is_index = index_finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y and \
                       index_finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y and \
                       index_finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y and \
                       index_finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            # Realizar clic si se ha extendido el dedo índice
            if is_index and not prev_state:
                pyautogui.click()

            # Guardar el estado anterior del dedo índice
            prev_state = is_index

    # Mostrar la imagen resultante
    cv2.imshow("Mouse Control", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
