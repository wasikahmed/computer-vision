from ultralytics import YOLO
import cv2
import os

# Load the model
model_path = os.path.join(os.getcwd(), '../yolo-Weights/yolo11m.pt')
model = YOLO(model=model_path)

# print(model.names)

# Load the image
image_path = os.path.join(os.getcwd(), 'cat_dog.jpg')
image = cv2.imread(image_path)

# Predict the results
results = model.predict(image_path)
result = results[0]
# print(result)

# Draw bounding boxes on the image
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list and unpack
    label = model.names[int(box.cls)]
    confidence = box.conf.item()  # Convert tensor to scalar
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()