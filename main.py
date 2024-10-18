from roboflow import Roboflow
rf = Roboflow(api_key="BhJ63IemG3ACRXXxUyzC")
project = rf.workspace("dental-decay").project("check-hfbzl-zemwx")
version = project.version(12)
dataset = version.download("yolov8")

from ultralytics import YOLO
import cv2
import os
import random
import shutil
import numpy as np
def preprocess_image(image_path, output_path):
    """
    این تابع برای اعمال فیلتر ها روی عکس انتخاب شده است
    به این صورت که ابتدا به طور رندوم یکی از ۵ متد ممکن
    انتخاب شده و بعد از انجام عملیات روی ان تطاویر انها
    در فایل مورد نظر ذخیره میشوند
    """
    img = cv2.imread(image_path)

    img = cv2.resize(img, (640, 640))

    random_filter = random.choice(['grayscale', 'brightness', 'contrast', 'noise', 'sharpen_edges'])

    if random_filter == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif random_filter == 'brightness':
        beta = random.uniform(0.5, 1.5)
        img = cv2.convertScaleAbs(img, alpha=1, beta=beta * 50)

    elif random_filter == 'contrast':
        alpha = random.uniform(1.0, 2.0)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    elif random_filter == 'noise':

        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    elif random_filter == 'sharpen_edges':

        edges = cv2.Canny(img, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)


    cv2.imwrite(output_path, img)


dataset_dir = "/content/CHECK-12/train/images" # دیتا ست اصلی برای اموزش
label_dir = "/content/CHECK-12/train/labels"  # دیتا ست لیبل های عکس های اصلی
augmented_dir = "/content/CHECK-12/train/images_augmented"  # دیتا ست عکس هایی که تغییر کردند
augmented_label_dir = "/content/CHECK-12/train/labels_augmented" #دیتا ست لیبل های عکس های تغییر یافته

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)
if not os.path.exists(augmented_label_dir):
    os.makedirs(augmented_label_dir)


images = os.listdir(dataset_dir)
random_images = random.sample(images, len(images) // 10)# به اندازه یک دهم از تعداد عکس های اصلی انتخاب میشوند برای بیش بردازش

dataset_dir = "/content/CHECK-12/train/images" # دیتا ست اصلی برای اموزش
label_dir = "/content/CHECK-12/train/labels"  # دیتا ست لیبل های عکس های اصلی
augmented_dir = "/content/CHECK-12/train/images_augmented"  # دیتا ست عکس هایی که تغییر کردند
augmented_label_dir = "/content/CHECK-12/train/labels_augmented" #دیتا ست لیبل های عکس های تغییر یافته

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)
if not os.path.exists(augmented_label_dir):
    os.makedirs(augmented_label_dir)


images = os.listdir(dataset_dir)
random_images = random.sample(images, len(images) // 10)# به اندازه یک دهم از تعداد عکس های اصلی انتخاب میشوند برای بیش بردازش


for img_name in random_images: #حلقه ای که ا طریق بسوند فایل هایی که میخواند لیبل های مورد نظر را جدا میکند
    img_path = os.path.join(dataset_dir, img_name)
    label_name = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, label_name)

    augmented_img_path = os.path.join(augmented_dir, img_name)
    augmented_label_path = os.path.join(augmented_label_dir, label_name)


    preprocess_image(img_path, augmented_img_path) # انجام تابع اعمال فیلتر ها برای ادرس هایی که تعریف کردیم


    shutil.copy(label_path, augmented_label_path)


for img_name in random_images: #اضافه کردن  عکس های تغییر یافته به مجموعه اصلی

    shutil.copy(os.path.join(augmented_dir, img_name), dataset_dir)
    shutil.copy(os.path.join(augmented_label_dir, img_name.replace('.jpg', '.txt')), label_dir)

    model = YOLO("/content/drive/MyDrive/best-90-cli-150im.pt")

    model.train(
        data="/content/CHECK-12/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        workers=3,
        cos_lr=True,
        momentum=0.9595,
        augment=True,
        lr0=1e-3,
        optimizer="Adam",

    )

    model.predict("/con