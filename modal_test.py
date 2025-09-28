# test_images_keypoint.py
# ใช้: python test_images_keypoint.py
# ติดตั้ง: pip install ultralytics opencv-python

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# ===== ตั้งค่า path ตรงนี้ =====
MODEL_PATH = r"C:\Users\ggwd4\Documents\keypoint2\runs\pose\train\weights\best.pt"
SOURCE     = r"C:\Users\ggwd4\Documents\keypoint2\Dog-Pose-2\test\images"  # โฟลเดอร์ภาพ หรือไฟล์ภาพเดี่ยว
OUT_DIR    = r"C:\Users\ggwd4\Documents\keypoint2\output"
# =================================

IMGSZ       = 640       # ใช้ให้ตรงกับตอนเทรน
CONF_BOX    = 0.50      # ค่าความมั่นใจขั้นต่ำสำหรับการตรวจจับ (กล่อง)
DEVICE      = 0         # 0=GPU ตัวแรก, หรือ "cpu"
WORKERS     = 0         # Windows-friendly

# === ค่าตกแต่ง label ===
DRAW_LABELS      = True     # เปิด/ปิดการวาง label ที่ keypoint
KP_CONF_MIN      = 0.25     # วาง label เฉพาะจุดที่ conf >= ค่านี้ (ถ้าโมเดลมี conf ต่อจุด)
SHOW_INDEX       = False    # แสดงหมายเลข index นำหน้าชื่อจุด
TEXT_THICKNESS   = 1
CIRCLE_RADIUS    = 2        # วาดจุดทับอีกทีให้เด่นขึ้นนิดหน่อย
BG_PADDING       = 2        # padding กล่องพื้นหลังข้อความ

KEYPOINT_NAMES = [
    "0","1","2","3","4",
    "5","6","7","8","9",
    "10","11","12","13","14",
    "15","16","17","18","19",
    "20","21","22","23","24","25"
]
def _auto_font_scale(img_w, img_h):
    # สเกลฟอนต์อิงกับขนาดภาพ เพื่อให้พอดีทุกความละเอียด
    return 0.4

def _put_label(img, text, org, font_scale, thickness):
    # วาดพื้นหลังสี่เหลี่ยม + ข้อความ เพื่อให้เห็นชัดทุกพื้นหลัง
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    # กล่องพื้นหลัง (สีดำโปร่ง ๆ)
    x1, y1 = x, y - th - baseline - BG_PADDING
    x2, y2 = x + tw + 2*BG_PADDING, y + BG_PADDING
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    # วาดพื้นหลังทึบก่อน (จะชัดที่สุด)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    # วาดตัวหนังสือ (สีขาว)
    cv2.putText(img, text, (x + BG_PADDING, y - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _draw_kp_labels(result, img):
    """
    วาด label ให้ keypoints ในภาพ img ตามผลลัพธ์ result ของ Ultralytics
    """
    if result.keypoints is None:
        return img

    kps_xy = result.keypoints.xy  # [num_person, K, 2]
    kps_conf = getattr(result.keypoints, 'confidence', None)  # อาจเป็น None ถ้าเวอร์ชัน/โมเดลไม่ให้ค่า
    kps_xy = kps_xy.cpu().numpy() if hasattr(kps_xy, "cpu") else np.asarray(kps_xy)
    if kps_conf is not None:
        kps_conf = kps_conf.cpu().numpy() if hasattr(kps_conf, "cpu") else np.asarray(kps_conf)

    H, W = img.shape[:2]
    font_scale = _auto_font_scale(W, H)

    for person_idx, person_kps in enumerate(kps_xy):
        # person_kps: [K, 2]
        for k_idx, (x, y) in enumerate(person_kps):
            if np.isnan(x) or np.isnan(y):
                continue
            # filter ตาม conf ของจุดถ้ามี
            if kps_conf is not None:
                if k_idx >= kps_conf.shape[1]:
                    pass  # กันพลาดกรณีรูปทรงผิด
                else:
                    if kps_conf[person_idx, k_idx] < KP_CONF_MIN:
                        continue

            # ชื่อจุด
            if k_idx < len(KEYPOINT_NAMES):
                name = KEYPOINT_NAMES[k_idx]
            else:
                name = f"KP_{k_idx}"

            label_text = f"{k_idx}: {name}" if SHOW_INDEX else name

            # วาดจุดทับอีกนิดให้เด่น
            cv2.circle(img, (int(x), int(y)), CIRCLE_RADIUS, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            # ขยับข้อความให้เยื้องจากจุด (ป้องกันทับกัน)
            tx = min(W - 1, int(x) + 4)
            ty = max(0, int(y) - 4)
            _put_label(img, label_text, (tx, ty), font_scale, TEXT_THICKNESS)

    return img

def main():
    # โหลดโมเดล
    model = YOLO(MODEL_PATH)

    # ถ้า SOURCE เป็นโฟลเดอร์ ให้แสดงจำนวนไฟล์คร่าว ๆ
    p = Path(SOURCE)
    if p.is_dir():
        n = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"})
        print(f"[INFO] พบรูป {n} ไฟล์ในโฟลเดอร์: {p}")

    # โฟลเดอร์ผลลัพธ์
    save_dir = Path(OUT_DIR) / "vis_labels"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ใช้ stream=True เพื่อวนทีละภาพ แล้วเราจะวาด label เอง
    results = model.predict(
        source=SOURCE,
        imgsz=IMGSZ,
        conf=CONF_BOX,
        device=DEVICE,
        workers=WORKERS,
        stream=True,          # สำคัญ: จะได้ผลลัพธ์ทีละไฟล์
        verbose=True
    )

    # ประมวลผลและบันทึก
    count = 0
    for result in results:
        # วาด keypoints/โครงกระดูกแบบ default ของ ultralytics ก่อน
        plotted = result.plot()  # ได้ภาพ overlay แล้ว

        # เติม label ที่ keypoints
        if DRAW_LABELS:
            plotted = _draw_kp_labels(result, plotted)

        # ตั้งชื่อไฟล์เอาต์พุต
        src_name = Path(result.path).stem  # ชื่อไฟล์ต้นทางไม่รวมนามสกุล
        out_path = save_dir / f"{src_name}_kp.jpg"

        # บันทึก
        cv2.imwrite(str(out_path), plotted)
        count += 1

    print("\n[DONE] รูปผลลัพธ์ถูกบันทึกไว้ที่:")
    print(save_dir)
    print(f"[INFO] บันทึก {count} ไฟล์")

if __name__ == "__main__":
    main()
