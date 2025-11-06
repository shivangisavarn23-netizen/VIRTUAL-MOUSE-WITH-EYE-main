import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False  # Disable failsafe to prevent corner crash

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    
    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0][0]), int(predicted[1][0])  # Extract values properly

kf = KalmanFilter()

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape
    
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        eye_x = int(landmarks[474].x * frame_w)
        eye_y = int(landmarks[474].y * frame_h)
        
        smooth_x, smooth_y = kf.predict(screen_w / frame_w * eye_x, screen_h / frame_h * eye_y)
        
        # Ensure cursor stays within screen bounds
        smooth_x = max(0, min(screen_w, smooth_x))
        smooth_y = max(0, min(screen_h, smooth_y))
        
        pyautogui.moveTo(smooth_x, smooth_y)
        
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y
        eye_distance = abs(left_eye_top - left_eye_bottom)
        
        if eye_distance < 0.003:
            pyautogui.click()
            cv2.putText(frame, "Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for idx in [474, 475, 476, 477]:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Right eye green
        
        for idx in [145, 159]:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Left eye red

    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
