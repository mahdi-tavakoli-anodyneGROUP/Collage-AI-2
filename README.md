
# **تشخیص ناحیه‌های سینوس در تصاویر OPG با استفاده از  مدل YOLOv8**

<h2 dir="rtl" align="right">Project Overview</h2>
<p dir="rtl" align="right">
این پروژه با هدف توسعه یک مدل یادگیری ماشین برای تشخیص نواحی سینوس در تصاویر OPG (Orthopantomogram) طراحی شده است. این مدل از YOLOv8 برای تشخیص و بخش‌بندی دقیق ناحیه سینوس استفاده می‌کند. مجموعه داده شامل 1802 تصویر برچسب‌گذاری شده است.
</p>

<h2 dir="rtl" align="right">Key Features</h2>
<p dir="rtl" align="right">
<ul>
<li>استفاده از معماری YOLOv8 برای تشخیص دقیق ناحیه‌های سینوس در تصاویر X-ray.</li>
<li>جمع‌آوری و آماده‌سازی داده‌ها با استفاده از Roboflow.</li>
<li>آموزش مدل روی مجموعه داده‌ای شامل 1802 تصویر OPG با برچسب‌های مخصوص نواحی سینوس.</li>
<li>بهینه‌سازی شده برای استنتاج سریع و دقت بالا در تصاویر پزشکی.</li>
</ul>
</p>

<h2 dir="rtl" align="right">Team Members</h2>
<p dir="rtl" align="right">
 مهندس کامیاب رفیعایی: برنامه‌نویس ارشد، مسئول انتخاب الگوریتم و بهینه‌سازی عملکرد مدل.<br>
 مهندس محدثه امیدوار: برنامه‌نویس، تمرکز بر بهینه‌سازی کد و افزایش کارایی آموزش مدل.
</p>


<h2 dir="rtl" align="right">Project Workflow</h2>
<ol dir="rtl" align="right">
<li><strong>جمع‌آوری داده‌ها و برچسب‌گذاری:</strong> <br>
یک مجموعه داده شامل 1802 تصویر OPG که هر تصویر به صورت دستی برای ناحیه‌های سینوس برچسب‌گذاری شده است.<br>
از Roboflow برای مدیریت مجموعه داده‌ها، برچسب‌ها و مراحل پیش‌پردازش استفاده شد.
</li>

<li><strong>آموزش مدل:</strong> <br>
مدل YOLOv8n.seg.pt به دلیل معماری سبک خود برای وظایف بخش‌بندی انتخاب شد. مدل با استفاده از Google Colab و تسریع GPU آموزش داده شد.<br>
مقادیر اصلی برای هایپرپارامترهای آموزشی:
<ul dir="rtl" align="right">
<li>Image size: 640x640</li>
<li>Number of epochs: 100</li>
<li>Batch size: 16</li>
<li>Workers: 2</li>
</ul>
مجموعه داده به سه بخش آموزشی، اعتبارسنجی و تست تقسیم شد تا ارزیابی و تعمیم صحیح مدل تضمین شود.
</li>

<li><strong>پیش‌بینی و استنتاج:</strong> <br>
پس از آموزش، مدل بر روی داده‌های تست اعمال شد و خروجی‌ها برای هر تصویر تست ذخیره شدند.
</li>
</ol>

<h2 dir="rtl" align="right">Code Explanation</h2>

### **Dataset Setup**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_xxx")
project = rf.workspace("xx-xx").project("xx-xx-xx")
version = project.version(12)
dataset = version.download("yolov8")
```
<p dir="rtl" align="right">
این بخش تنظیمات مربوط به مجموعه داده‌ها را با استفاده از API Roboflow انجام می‌دهد و داده‌های آماده‌شده و برچسب‌گذاری شده را دانلود می‌کند.
</p>

### **Configuration and Data Preparation**
```python
import yaml

with open("/content/CHECK-12/data.yaml", 'r') as f:
    dataset_yaml = yaml.safe_load(f)
dataset_yaml["train"] = "../train/images"
dataset_yaml["val"] = "../valid/images"
dataset_yaml["test"] = "../test/images"
with open('/content/CHECK-12/data.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)
```
<p dir="rtl" align="right">
فایل `data.yaml` به‌روزرسانی می‌شود تا به مسیرهای صحیح برای مجموعه‌های آموزشی، اعتبارسنجی و تست اشاره کند.
</p>

### **Model Training**
```python
from ultralytics import YOLO
model = YOLO('yolov8m-seg.pt')
model.train(data='/content/CHECK-12/data.yaml', imgsz=640, epochs=150, batch=16, workers=2)
```
<p dir="rtl" align="right">
YOLOv8 نصب شده و برای آموزش مدل استفاده می‌شود. مدل برای 150 دوره با اندازه دسته 16 و اندازه ورودی 640x640 پیکسل آموزش داده می‌شود.
</p>

### **Model Inference**
```python
import glob

folder_path = '/content/CHECK-12/test/images'
image_paths = glob.glob(f"{folder_path}/*.jpg")  # You can include png or any other formats

for image_path in image_paths:
    model.predict(image_path, save=True)
```
<p dir="rtl" align="right">
این بخش تست مدل را اجرا می‌کند و پیش‌بینی‌ها برای هر تصویر تست ذخیره می‌شوند.
</p>

<h2 dir="rtl" align="right">Usage</h2>
<p dir="rtl" align="right">
1. ابتدا مجموعه داده‌ها را با استفاده از Roboflow دانلود کنید.<br>
2. سپس از کد آموزش مدل با YOLOv8 برای تشخیص ناحیه سینوس استفاده کنید.
</p>

<h2 dir="rtl" align="right">License</h2>
<p dir="rtl" align="right">
این پروژه توسط شرکت Anodyne  منتشر شده است. 
</p>
