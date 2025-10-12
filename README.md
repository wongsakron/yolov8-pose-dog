# 🐶 AI Dog Pose Detection App
แอปตรวจจับท่าทางสุนัขด้วย YOLOv8 + Roboflow + Gradio  
สร้างส่วนติดต่อใช้งานผ่านเว็บ (Web UI) สำหรับอัปโหลดรูปหรือเปิดกล้องเพื่อทดสอบโมเดล AI  

---

## 📦 Project Structure

```
project/
│
├── app.py                # Main file สำหรับรัน Gradio App
├── set_modal.py          # โหลดและตั้งค่า YOLO model
├── requirements.txt      # รายชื่อไลบรารีทั้งหมดที่ต้องใช้
├── .env                  # (ทางเลือก) เก็บ API key ของ Roboflow
└── README.md             # ไฟล์อธิบายโปรเจกต์ (ไฟล์นี้)
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

หรือถ้าไม่มีไฟล์ requirements.txt  
ให้ใช้คำสั่งติดตั้งแบบรวดเดียว:
```bash
pip install numpy opencv-python pillow torch torchvision torchaudio ultralytics roboflow gradio python-dotenv
```

---

## 🧩 Requirements.txt (ตัวอย่าง)

```text
numpy
opencv-python
pillow
torch
torchvision
torchaudio
ultralytics
roboflow
gradio
python-dotenv
```

---

## 🚀 Running the App

### 1️⃣ รันโมเดล YOLO ผ่าน Gradio
```bash
python app.py
```

เมื่อรันสำเร็จ ระบบจะแสดงลิงก์ใน Terminal เช่น  
```
Running on local URL:  http://127.0.0.1:7860
```
เปิดในเว็บเบราว์เซอร์เพื่อทดลองอัปโหลดภาพหรือเปิดกล้องได้เลย 🎥  

---

## 🧠 About the Model

- ใช้ **YOLOv8 (Ultralytics)** สำหรับตรวจจับ keypoints หรือท่าทางของสุนัข  
- โหลด Dataset/Model ผ่าน **Roboflow API** (ตั้งค่าใน `.env`)  
- แสดงผลผ่าน **Gradio Interface**

---

## 🌱 Example .env File

```env
ROBOFLOW_API_KEY=your_api_key_here
MODEL_VERSION=1
PROJECT_NAME=dog-pose-detection
```

---

## 🖼️ Example Output

| Input Image | Detection Result |
|--------------|------------------|
| ![dog1](https://via.placeholder.com/200x150.png?text=Dog+Image) | ![result](https://via.placeholder.com/200x150.png?text=Pose+Detected) |

---

## 💻 Tech Stack

| Component | Description |
|------------|-------------|
| **Python 3.10+** | ภาษาหลักของโปรเจกต์ |
| **Gradio** | สร้าง Web UI สำหรับรันโมเดล |
| **Ultralytics YOLOv8** | ตรวจจับวัตถุและ keypoints |
| **Roboflow** | ใช้จัดการ Dataset / API key |
| **OpenCV** | จัดการภาพและกล้อง |
| **Pillow (PIL)** | แสดงผลภาพและเขียนข้อความบนภาพ |

---

## 🧑‍💻 Developer

👤 **หมี่เกี๊ยว**  
🎓 Walailak University — ASEAN Studies × IT Entrepreneur  
💡 Passion: AI, Localization, Soft Power Research  

---

## ⭐ License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute.

---

## 💬 Support
หากต้องการความช่วยเหลือเพิ่มเติม  
เปิด Issue หรือถามได้ใน Discussion ของ Repo ❤️
