# 🐶 YOLOv8 Dog Pose — Train & Web Inference (Separated)

โปรเจกต์นี้แบ่ง **สองส่วนการทำงานชัดเจน**:
1) **Training** — ทำใน `main.py` (โหลด dataset และเทรนโมเดล)  
2) **Web Inference** — ทำใน `Gradio/app_gradio.py` (ใช้งานโมเดลที่เทรนเสร็จแล้วผ่านเว็บ)

---

## 📦 Project Structure

```
yolov8-pose-dog-main/
│
├── yolov8n-pose.pt           # Base model ของ YOLOv8 Pose (ใช้เป็นจุดเริ่มต้นตอนเทรน)
├── .env                      # ตัวแปรสภาพแวดล้อม เช่น ROB0FLOW_API_KEY, PROJECT, VERSION ฯลฯ
├── main.py                   # สคริปต์ "เทรน" โมเดล (โหลด dataset + train + export)
├── modal_test.py             # โค้ดทดลอง/ดีบักโมเดล (ออปชัน)
├── README.md                 # ไฟล์นี้
│
├── Gradio/                   # ส่วน "เว็บ" สำหรับใช้งานโมเดลหลังเทรนเสร็จ
│   ├── app_gradio.py         # รันเว็บ Gradio เพื่ออัปโหลดภาพ/วิดีโอ/กล้อง แล้วให้โมเดลทำนาย
│   ├── set_modal.py          # โหลด weights (best.pt) + วาด keypoints/ผลลัพธ์ลงภาพ
│   ├── best.pt               # **ไฟล์โมเดลที่เทรนเสร็จ** (คัดลอกมาจาก runs/pose/train/weights/best.pt)
│   └── __pycache__/
│
├── runs/                     # โฟลเดอร์ผลลัพธ์การเทรน (Ultralytics สร้างให้อัตโนมัติ)
│   └── pose/
│       └── train/            # logs, results.png, confusion_matrix.png, weights/{best.pt,last.pt}
│
└── (ไฟล์อื่น ๆ เช่น data.yaml, requirements.txt)
```

> หมายเหตุ: โฟลเดอร์ `Gradio/` จะไม่ใช้สำหรับเทรน ใช้แค่รันเว็บ **หลัง** เทรนเสร็จเท่านั้น

---

## ⚙️ Installation

### 1) สร้าง Virtual Environment (แนะนำ)
```bash
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2) ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```
หรือถ้าไม่มี `requirements.txt` ให้ใช้:
```bash
pip install numpy opencv-python pillow torch torchvision torchaudio ultralytics roboflow gradio python-dotenv
```

---

## 🔐 Environment Variables (`.env` ตัวอย่าง)

```env
# ถ้าใช้ Roboflow สำหรับดาวน์โหลด/โหลด dataset หรือ weights
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=dog-pose-detection
ROBOFLOW_VERSION=1

# ตัวเลือกอื่นที่สคริปต์ main.py อาจอ่านได้
DATA_YAML=./data.yaml
BASE_MODEL=yolov8n-pose.pt
```

> ตั้งค่าตามที่ `main.py` ของโปรเจกต์อ่านใช้งานจริง

---

## 🏋️ Training Workflow (ทำใน root ด้วย `main.py`)

1. ตรวจสอบ/เตรียม `.env` และไฟล์ `data.yaml` ให้ถูกต้อง  
2. เริ่มเทรน (ตัวอย่างคำสั่ง สมมติว่า `main.py` รองรับพารามิเตอร์เหล่านี้):
```bash
python main.py --epochs 100 --imgsz 640 --batch 16 --model yolov8n-pose.pt
# หรือถ้า main.py ดึง dataset จาก Roboflow อัตโนมัติ ก็แค่เรียก
python main.py
```
3. หลังจบการเทรน ไฟล์ weights จะอยู่ที่:
```
runs/pose/train/weights/best.pt
```
4. คัดลอกไฟล์ `best.pt` ไปที่โฟลเดอร์เว็บ:
```
cp runs/pose/train/weights/best.pt Gradio/best.pt
```

---

## 🌐 Web Inference (ทำใน `Gradio/` หลังเทรนเสร็จ)

1. ย้ายไปที่โฟลเดอร์เว็บ
```bash
cd Gradio
```
2. ยืนยันว่ามี `best.pt` อยู่ในโฟลเดอร์นี้แล้ว (จากขั้นตอนคัดลอกด้านบน)  
3. รันเว็บ
```bash
python app_gradio.py
```
4. เปิด URL ที่ขึ้นใน Terminal เช่น
```
Running on local URL:  http://127.0.0.1:7860
```
แล้วใช้งานเว็บเพื่ออัปโหลดภาพ/วิดีโอหรือเปิดกล้องเพื่อทดสอบโมเดล

---

## 🧠 Notes & Tips

- ถ้าจะเทรนใหม่อีกครั้ง ให้ลบ/เปลี่ยนชื่อโฟลเดอร์ `runs/pose/train/` หรือใช้พารามิเตอร์ `project`/`name` ของ Ultralytics เพื่อแยกไดเรกทอรีผลลัพธ์
- ผลลัพธ์สำคัญหลังเทรน:  
  - `runs/pose/train/weights/best.pt` (โมเดลที่ดีที่สุด)  
  - `runs/pose/train/results.png` (กราฟสรุปผลเทรน)  
- หากใช้งานกล้องแล้วมีปัญหา `openh264` บน Windows ให้ติดตั้ง/อัปเดต OpenH264 หรือใช้การทำนายจากภาพนิ่งแทนชั่วคราว

---

## ✅ Requirements (อ้างอิง)

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

## 📄 License

MIT License
