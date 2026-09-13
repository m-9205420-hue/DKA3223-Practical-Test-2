
from ultralytics import YOLO

# YOLO can detect objects and their locations using bounding boxes,
# while a basic ANN mainly performs classification without spatial information.

# YOLO preserves spatial information from the image,
# which is important for locating objects.

# YOLO can detect multiple objects in a single image,
# while a basic ANN is not designed to directly locate multiple objects.

# YOLO is suitable for real-time object detection because detection
# is performed efficiently in a single model pipeline.

# Load YOLOv8 nano pre-trained model
model = YOLO("yolov8n.pt")

# Run object detection with the required confidence and IoU thresholds
results = model(
    "gmbqkeretamoto.png",
    conf=0.40,
    iou=0.50
)

# Save the detection result
results[0].save(filename="result_traffic.jpg")

print("Detection completed successfully!")
print("Output saved as result_traffic.jpg")
