# 🐶 YOLOv8 Dog Pose Detection

โปรเจกต์นี้ใช้ **YOLOv8 Pose Model** สำหรับตรวจจับท่าทางของสุนัข (keypoints detection)  
พร้อม Web Interface ที่สร้างด้วย **Gradio** เพื่อให้ผู้ใช้สามารถอัปโหลดภาพหรือเปิดกล้องทดสอบโมเดลได้ง่าย ๆ  

---

## 📦 Project Structure

```
yolov8-pose-dog-main/
│
├── yolov8n-pose.pt           # โมเดล YOLOv8 Pose (base model)
├── .env                      # เก็บ API Key / environment variables
├── main.py                   # สคริปต์หลักสำหรับทดสอบหรือรันโมเดล
├── modal_test.py             # สคริปต์ทดลองโมเดล (ใช้ปรับแต่ง parameter / visualize)
├── README.md                 # ไฟล์คำอธิบายโปรเจกต์ (ไฟล์นี้)
│
├── Gradio/                   # ส่วนของ Gradio Web App
│   ├── app_gradio.py         # สคริปต์หลักของเว็บแอป (เปิด Gradio interface)
│   ├── set_modal.py          # โหลดและตั้งค่า YOLO model + วาด keypoints บนภาพ
│   ├── best.pt               # โมเดลที่เทรนเสร็จและเลือกมาใช้จริง (best weights)
│   └── __pycache__/          # Cache ของ Python
│
├── runs/                     # โฟลเดอร์เก็บผลการเทรน YOLO (สร้างอัตโนมัติ)
│   └── pose/
│       └── train/            # Log, model weights, result images จากการเทรน
│
└── (ไฟล์อื่น ๆ เช่น dataset.yaml, requirements.txt)
```

---

## ⚙️ Installation

### 1️⃣ สร้าง Virtual Environment (แนะนำ)
```bash
python -m venv venv
source venv/bin/activate       # (Mac/Linux)
venv\Scripts\activate          # (Windows)
```

### 2️⃣ ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```
หรือหากไม่มีไฟล์ `requirements.txt` ให้ใช้คำสั่งนี้:
```bash
pip install numpy opencv-python pillow torch torchvision torchaudio ultralytics roboflow gradio python-dotenv
```

---

## 🚀 Running the App

### ✅ 1. รันผ่าน Gradio UI
```bash
cd Gradio
python app_gradio.py
```
จากนั้นเปิด URL ที่แสดงใน Terminal เช่น  
```
Running on local URL:  http://127.0.0.1:7860
```

### ✅ 2. ทดสอบโมเดลตรงผ่าน Terminal
```bash
python main.py --source path/to/image.jpg
```

---

## 🌱 Example .env File

```env
ROBOFLOW_API_KEY=your_api_key_here
MODEL_VERSION=1
PROJECT_NAME=dog-pose-detection
```

---

## 🧠 About the Model

- ใช้ **YOLOv8 (Ultralytics)** สำหรับตรวจจับ keypoints ของสุนัข  
- สามารถเทรนเองหรือนำโมเดลจาก **Roboflow** มาใช้งานได้  
- รองรับการทดสอบแบบ real-time ผ่านกล้องเว็บแคม  

---

## 💻 Tech Stack

| Component | Description |
|------------|-------------|
| **Python 3.10+** | ภาษาโปรแกรมหลัก |
| **Ultralytics YOLOv8** | โมเดลตรวจจับวัตถุและ keypoints |
| **Gradio** | Web UI สำหรับอัปโหลดภาพและทดสอบโมเดล |
| **OpenCV** | จัดการภาพและกล้อง |
| **Pillow (PIL)** | วาด keypoints / ข้อความลงบนภาพ |
| **Roboflow** | ใช้จัดการ Dataset และ API |
| **Dotenv** | โหลดค่า environment เช่น API Key |

---

## 🖼️ Example Output

| Input Image | Detection Result |
|--------------|------------------|
| ![dog1](https://via.placeholder.com/200x150.png?text=Dog+Image) | ![result](https://via.placeholder.com/200x150.png?text=Pose+Detected) |

---

## 🧩 Tips

- หากต้องการเทรนโมเดลใหม่ ให้ใช้คำสั่งจาก Ultralytics เช่น  
  ```bash
  yolo pose train data=data.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
  ```
- โมเดลที่เทรนเสร็จจะถูกบันทึกไว้ใน `runs/pose/train/weights/best.pt`

---

## ⭐ License

This project is licensed under the **MIT License** — You are free to use, modify, and distribute.

---

## 💬 Support
สามารถเปิด Issue ใน GitHub Repository ได้หากต้องการรายงานปัญหาหรือแนะนำเพิ่มเติม
