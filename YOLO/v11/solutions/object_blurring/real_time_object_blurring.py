import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

# Variables to calculate FPS
prev_time = 0
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Predict and blur objects
    results = model.predict(frame, show=False)
    for box in results[0].boxes.xyxy.cpu().tolist():
        obj = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
        frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = cv2.blur(obj, (50, 50))
    
    # Overlay FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("YOLO11 Blurring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
