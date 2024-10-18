from roboflow import Roboflow
rf = Roboflow(api_key="BhJ63IemG3ACRXXxUyzC")
project = rf.workspace("dental-decay").project("alpha-1-60hzc")
version = project.version(3)
dataset = version.download("yolov8-obb")
from ultralytics import YOLO
model = YOLO("/content/runs/segment/train22/weights/best.pt")
model.train(
    data = "/content/alpha-1-3/data.yaml" ,
    epochs =200 ,
    imgsz = 640 ,
    batch = 18  ,
    workers = 3 ,
    cos_lr = True ,
    momentum = 0.9595 ,
    augment = True ,
    lr0 = 1e-3 ,
    optimizer = "Adam" ,
    patience = 20

)
model.predict("/content/alpha-1-3/test/images" , save = True ,conf = 0.7 )