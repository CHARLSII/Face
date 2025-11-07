from roboflow import Roboflow
from ultralytics import YOLO

# ✅ Step 1: Connect to Roboflow
rf = Roboflow(api_key="j1lsR6Qm2ufivlsEEBHa")
project = rf.workspace("charls-lab").project("face-recognition-lrmfu")
version = project.version(5)
dataset = version.download("yolov8")
                

# ✅ Step 2: Train the model
# The dataset.path automatically points to the YOLO-formatted dataset folder
model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt, yolov8m.pt, etc.

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,        # you can increase for better accuracy
    imgsz=320,        # image size
    batch=8,         # batch size
    name="face_recognition_run",
)

# ✅ After training, your best model weights will be saved in:
# runs/detect/face_recognition_run/weights/best.pt
print("Training complete! The best model is saved at: runs/detect/face_recognition_run/weights/best.pt")
