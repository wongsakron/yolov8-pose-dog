import os
import sys
import time
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# (ทางเลือก) โหลดค่า ENV จากไฟล์ .env ถ้ามี
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Config & Helpers ----------
SEED = 42                                # ตั้งค่า random seed เพื่อให้ผล reproducible
DEVICE = 0 if torch.cuda.is_available() else 'cpu'   # เลือก GPU ถ้ามี ไม่งั้นใช้ CPU
torch.manual_seed(SEED)                  # set seed สำหรับ CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)     # set seed สำหรับ GPU ด้วย

def env_required(name: str) -> str:
    """ฟังก์ชันเล็ก ๆ ไว้บังคับว่า ENV ตัวนี้ต้องมี ไม่งั้น throw error"""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

# โหลดค่าจาก ENV (ต้องไปตั้งค่าไว้ในระบบหรือไฟล์ .env)
RF_API_KEY  = env_required("RF_API_KEY")        # API Key ของ Roboflow
RF_WORKSPACE = env_required("RF_WORKSPACE")     # ชื่อ workspace ใน Roboflow
RF_PROJECT   = env_required("RF_PROJECT")       # ชื่อ project
RF_VERSION   = int(os.getenv("RF_VERSION", "1"))# เวอร์ชัน dataset (default = 1)

# ค่าอื่น ๆ ที่ตั้งผ่าน ENV หรือใช้ default
MODEL_NAME = os.getenv("POSE_MODEL")  # โมเดลเริ่มต้น 
IMGSZ = int(os.getenv("IMGSZ"))                   # ขนาดภาพ
BATCH = int(os.getenv("BATCH"))                    # batch size (จะ auto ลดถ้า OOM)
EPOCHS = int(os.getenv("EPOCHS"))                  # จำนวน epoch รวม (เราจะแบ่งเป็น 2 เฟส)

print(f"Device: {DEVICE}")

# ---------- Roboflow download (skip if already present) ----------
rf = Roboflow(api_key=RF_API_KEY)                # login ด้วย API Key
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT) # เลือก workspace/project
version = project.version(RF_VERSION)            # เลือกเวอร์ชัน dataset

# path โฟลเดอร์ dataset (เช็กว่ามีอยู่แล้วหรือยัง)
target_dir = os.path.join(os.getcwd(), f"{RF_PROJECT}-{RF_VERSION}")
data_yaml = os.path.join(target_dir, "data.yaml")

# ถ้าเจอ data.yaml แล้ว แสดงว่าโหลดไว้แล้ว → skip download
if os.path.exists(data_yaml):
    print("Found existing dataset, skip download ->", data_yaml)
else:
    ds = version.download("yolov8")              # โหลดใหม่ในฟอร์แมต YOLOv8
    target_dir = ds.location
    data_yaml = os.path.join(target_dir, "data.yaml")
    print("Downloaded dataset ->", data_yaml)

# ---------- Small OOM-safe trainer ----------
def try_train(model, **kwargs):
    """
    รัน train แบบกัน OOM: ถ้าเจอ CUDA out of memory จะลด batch → ลด imgsz แล้วลองใหม่อัตโนมัติ
    """
    batch = int(kwargs.get("batch", BATCH))
    imgsz = int(kwargs.get("imgsz", IMGSZ))
    while True:
        try:
            print(f"[train] epochs={kwargs.get('epochs')} batch={batch} imgsz={imgsz}")
            return model.train(**{**kwargs, "batch": batch, "imgsz": imgsz})
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                if batch > 4:
                    batch = max(4, batch // 2)
                    print(f"[OOM] ลด batch → {batch} แล้วลองใหม่...")
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    continue
                elif imgsz > 512:
                    imgsz = 512
                    print(f"[OOM] ลด imgsz → {imgsz} แล้วลองใหม่...")
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    continue
            raise  # ถ้าเป็น error อื่นให้โยนต่อ

# ---------- Phase 1: Base Fit (เรียนรู้ภาพรวม) ----------
# - aug ปานกลาง, lr เริ่มต้น, patience สูงขึ้นเล็กน้อย
phase1_epochs =EPOCHS  # 20–40
model = YOLO(MODEL_NAME)                  # โหลดโมเดลฐาน เช่น yolov8s-pose
res1 = try_train(
    model,
    data=data_yaml,                               # data.yaml ระบุ class=dog + keypoints schema
    epochs=phase1_epochs,
    imgsz=IMGSZ,                                    # ภาพอินพุต 640x640
    device=DEVICE,
    workers=0,                                    # Windows ให้ 0
    optimizer="adamw",                            # AdamW เหมาะกับ keypoint
    lr0=3e-4,                                     # ลด LR เล็กน้อย เพราะ dataset เล็ก/ซับซ้อน
    lrf=0.01,
    patience=8,                                   # รอ early stopping นานขึ้น เพราะ loss อาจแกว่ง
    pretrained=True,                              # เริ่มจาก pretrained weights

      # --- Augment สำหรับ dog pose (นุ่มนวล) ---
    hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
    degrees=10, translate=0.05, scale=0.15,
    shear=0.0, mosaic=0.0,
    perspective=0.0,                              # ไม่บิดภาพ
    fliplr=0.3,
    val=True,
    cache=True,                                 # เร่งโหลด dataset
)
final_best = os.path.join(res1.save_dir, "weights", "best.pt")
print("Phase 1 best:", final_best)


# รัน validate อีกรอบ เพื่อดู OKS/mAP/PCK
final_model = YOLO(final_best)
val_res = final_model.val(data=data_yaml, device=DEVICE, imgsz=IMGSZ)
print(val_res)

