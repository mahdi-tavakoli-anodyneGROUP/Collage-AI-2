from roboflow import Roboflow

rf = Roboflow(api_key="BhJ63IemG3ACRXXxUyzC")
project = rf.workspace("dental-decay").project("alpha-1-60hzc")
version = project.version(3)
dataset = version.download("yolov8-obb")
from ultralytics import YOLO
from google.colab import drive
drive.mount('/content/drive')
model = YOLO("yolov8n-seg.pt")
model.train(
    data="/content/CHECK-12/data.yaml",
    epochs=50,
    imgsz=640,
    batch=12,

)

res = model.predict(source="/content/alpha-1/test/images", save=True)
