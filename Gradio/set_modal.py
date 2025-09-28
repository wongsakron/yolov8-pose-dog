
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
# ===== ตั้งค่าโมเดล =====
MODEL_PATH = "best.pt"  # เปลี่ยน path ตามเครื่องหมี่เกี๊ยว

# เกณฑ์เริ่มต้นสำหรับความมั่นใจของ 'จุด'
MIN_KP_CONF_DEFAULT=0.30 # = 30%
# ขนาดพท้นหลังฟอนต์'
BG_PADDING = 3
CIRCLE_RADIUS = 2
# ===== ฟอนต์ไทย (แก้ให้ชี้ฟอนต์ที่มีในเครื่องได้) =====
# TH Sarabun New / Noto Sans Thai
FONT_PATH_CANDIDATES = [
    r"C:\Windows\Fonts\angsana.ttc",
    r"C:\Windows\Fonts\angsanau.ttf"
]
def _pick_font_path():
    for p in FONT_PATH_CANDIDATES:
        if Path(p).exists():
            return p
    return None
# ขนาดฟอนต์'
FONT_PATH = _pick_font_path()
FONT_SIZE_BASE = 22  # ขนาดเบส (จะคูณสเกลอีกที)


# ===== รายชื่อจุด (0–25) ภาษาไทย =====
KEYPOINT_NAMES = [
    "อุ้งเท้าหน้าขวา ", "ข้อมือหน้าขวา ", "ไหล่ขวา", "อุ้งเท้าหลังขวา", "ข้อเท้าหลังขวา",
    "สะโพกขวา", "อุ้งเท้าหน้าซ้าย", "ข้อมือหน้าซ้าย", "ไหล่ซ้าย", "อุ้งเท้าหลังซ้าย",
    "ข้อเท้าหลังซ้าย", "สะโพกซ้าย", "โคนหาง", "ปลายหาง", "หูขวา", "หูซ้าย",
    "จมูก", "ปาก", "หูขวา", "หูซ้าย", "ท้อง/เอว", "Null",
    "บ่าซ้าย", "ท้องซ้าย", "หลังล่างขวา", "กลางลำตัว"
]



def _auto_font_scale(w, h):
    # ย่อ/ขยายตามขนาดภาพ
    return 0.35

def _get_font(scale):
    size_px = max(12, int(FONT_SIZE_BASE * scale))
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size_px)
        except Exception:
            pass
    return ImageFont.load_default()



# ===== ปรับฟังก์ชันวาดฉลากให้พื้นหลังดำอยู่หลังข้อความแบบตรงเป๊ะ =====
def _put_label_pil(img_bgr, text, anchor_xy, font_scale):
    """
    วาดกล่องดำ + ข้อความไทย (ใช้ PIL) แล้วคืน BGR
    anchor_xy = พิกัดมุมซ้ายบนของข้อความ (top-left)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_scale)

    # ยึด anchor เป็น top-left ของตัวอักษร
    x, y = int(anchor_xy[0]), int(anchor_xy[1])

    # คำนวณกรอบข้อความตามตำแหน่งจริง (ไม่ลบ th อีกแล้ว)
    x1, y1, x2, y2 = draw.textbbox((x, y), text, font=font)

    # เพิ่ม padding รอบข้อความ
    x1 -= BG_PADDING
    y1 -= BG_PADDING
    x2 += BG_PADDING
    y2 += BG_PADDING

    # กันหลุดขอบภาพ
    W, H = pil_img.size
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)

    # พื้นหลังสีดำ
    draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0))
    # ข้อความสีขาววางพิกัดเดิม (top-left)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# โหลดโมเดลครั้งเดียว (เริ่มด้วย CPU; ถ้ามี CUDA จะย้ายให้)
model = YOLO(MODEL_PATH)
try:
    model.to(0)  # 'cuda:0'
except Exception:
    pass  # ไม่มี GPU ก็วิ่งบน CPU ได้


def infer(image: Image.Image, conf: float, show_index: bool, min_kp_conf: float = MIN_KP_CONF_DEFAULT):
    if image is None:
        return None, None

    img_rgb = np.array(image.convert("RGB"))
    results = model.predict(img_rgb, conf=conf, verbose=False)
    if not results or results[0] is None:
        return None, None
    r = results[0]

    try:
        plotted = r.plot()  # BGR
    except Exception:
        plotted = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)

    H, W = plotted.shape[:2]
    fscale = _auto_font_scale(W, H)
    rows = []

    if getattr(r, "keypoints", None) is None:
        img_out = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        headers = ["บุคคล", "ดัชนีจุด", "ชื่อจุด (ไทย)", "x", "y", "conf"]
        table = [headers] + rows if rows else None
        return img_out, table

    # ----- ดึงพิกัดและความมั่นใจของ 'จุด' -----
    kps_xy = r.keypoints.xy  # [N,K,2] (torch or np)
    kps_conf = getattr(r.keypoints, "conf", None)  # <= ใช้ conf เท่านั้นในรุ่นนี้

    kps_xy = kps_xy.cpu().numpy() if hasattr(kps_xy, "cpu") else np.asarray(kps_xy)
    if kps_conf is not None:
        kps_conf = kps_conf.cpu().numpy() if hasattr(kps_conf, "cpu") else np.asarray(kps_conf)

    def _get_conf(pi, ki):
        if kps_conf is None:
            return None
        val = kps_conf[pi, ki]
        try:
            val = float(np.squeeze(val))   # รองรับ [N,K] หรือ [N,K,1]
        except Exception:
            return None
        if np.isnan(val):
            return None
        return val

    N, K = kps_xy.shape[:2]
    for pi in range(N):
        for ki in range(K):
            x, y = kps_xy[pi, ki, 0], kps_xy[pi, ki, 1]
            if np.isnan(x) or np.isnan(y):
                continue

            name = KEYPOINT_NAMES[ki] if ki < len(KEYPOINT_NAMES) else f"จุด {ki}"
            if name.strip().lower() == "null":
                continue

            conf_k = _get_conf(pi, ki)

            # กฎ: conf_k เป็น None/NaN หรือ < min_kp_conf -> ไม่วาด
            if (conf_k is None) or (float(conf_k) < float(min_kp_conf)):
                continue

            cv2.circle(plotted, (int(x), int(y)), CIRCLE_RADIUS, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            label_text = f"{ki}: {name}" if show_index else name

            tx = min(W - 1, int(x) + 6)
            ty = max(0, int(y) - 6)
            plotted = _put_label_pil(plotted, label_text, (tx, ty), fscale)

            rows.append([int(pi), int(ki), name, round(float(x), 2), round(float(y), 2), round(float(conf_k), 3)])

    img_out = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    headers = ["บุคคล", "ดัชนีจุด", "ชื่อจุด (ไทย)", "x", "y", "conf"]
    table = [headers] + rows if rows else None
    return img_out, table
