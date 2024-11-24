from ultralytics import YOLO
import cv2
import os
import sys

# Load the model
model_path = os.path.join(os.getcwd(), '../yolo-Weights/yolo11m.pt')
model = YOLO(model=model_path)

# print(model.names)

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])
    
source = cv2.VideoCapture(s)

win_name = 'YOLOx11 Basic Detection'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while(cv2.waitKey(1) != 27): # 27 is the ASCII code for the ESC key
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb)
    result = results[0]
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[int(box.cls)]
        confidence = box.conf.item()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyAllWindows()