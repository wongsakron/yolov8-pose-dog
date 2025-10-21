# app.py
import os, csv, time, shutil, subprocess
import uuid
import cv2
import numpy as np
import gradio as gr

from set_modal import (
    infer,            # ใช้กับภาพนิ่ง
    infer_frame_bgr,  # ใช้ประมวลผลทีละเฟรมของวิดีโอ
    MIN_KP_CONF_DEFAULT
)

# ---------------- Utility ----------------
def _get_video_path(video):
    """ดึง path จริงของไฟล์วิดีโอจากอินพุต Gradio (รองรับหลายรูปแบบ)"""
    if video is None:
        return None
    if isinstance(video, str):
        return video if os.path.exists(video) else None
    if isinstance(video, dict):
        for k in ("path", "name", "tempfile", "file"):
            p = video.get(k)
            if isinstance(p, str) and os.path.exists(p):
                return p
    for attr in ("name", "path"):
        p = getattr(video, attr, None)
        if isinstance(p, str) and os.path.exists(p):
            return p
    return None

def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def _ffmpeg_h264_writer(out_stub, fps, width, height):
    """สร้างไฟล์ MP4 ด้วย H.264 ที่เล่นได้บนเบราว์เซอร์"""
    fps = max(1.0, float(fps))
    out_mp4 = out_stub + ".mp4"
    # บังคับให้กว้าง/สูงเป็นเลขคู่และใช้ yuv420p เพื่อความเข้ากันได้
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc, proc.stdin, out_mp4


# ---------------- App (UI แบบเวอร์ชันแรก + วิดีโอมีเอาต์พุตเดียว) ----------------
with gr.Blocks(title="Dog Pose (Thai Labels) – YOLO + Gradio") as app:
    gr.Markdown("## 🐶 Dog Pose Estimation (Thai Labels)\nอัปโหลดภาพหมาหรือวิดีโอ แล้วระบบจะทำนาย keypoints พร้อมป้ายชื่อภาษาไทย")

    history = gr.State([])

    # ========== UI: ภาพนิ่ง ==========
    with gr.Tab("ภาพนิ่ง"):
        with gr.Row():
            inp = gr.Image(type="pil", label="อัปโหลดภาพ", container=True)
            out_img = gr.Image(type="numpy", label="ผลลัพธ์", interactive=False)

        with gr.Row():
            conf = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="ค่าความมั่นใจขั้นต่ำ (conf)")
            show_index = gr.Checkbox(value=False, label="โชว์เลขดัชนี (0–25) ข้างชื่อจุด")

        run_btn = gr.Button("ทำนาย (ภาพนิ่ง)")

        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                with gr.Accordion("📜 ประวัติการทำนาย", open=False):
                    gallery = gr.Gallery(
                        label=None,
                        columns=3,
                        height=200,
                    )
            with gr.Column(scale=1):
                pass

        def predict_and_store(image, conf, show_index, hist):
            out_np, _ = infer(image, conf, show_index)
            if out_np is not None:
                hist = (hist or []) + [out_np]
                if len(hist) > 30:
                    hist = hist[-30:]
            return out_np, gr.update(value=hist), hist

        run_btn.click(
            fn=predict_and_store,
            inputs=[inp, conf, show_index, history],
            outputs=[out_img, gallery, history],
        )

        inp.upload(
            fn=predict_and_store,
            inputs=[inp, conf, show_index, history],
            outputs=[out_img, gallery, history],
        )

    # ========== UI: วิดีโอ ==========
    with gr.Tab("วิดีโอ"):
        with gr.Row():
            in_vid = gr.Video(label="อัปโหลดวิดีโอ", sources=["upload"], interactive=True)
            out_vid = gr.Video(
                label="วิดีโอผลลัพธ์ (ทำนายทั้งคลิป, MP4/H.264)",
                interactive=False,
                autoplay=True,
                show_download_button=True,
            )

        with gr.Row():
            conf_v = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="ค่าความมั่นใจขั้นต่ำ (conf)")
            show_index_v = gr.Checkbox(value=False, label="โชว์เลขดัชนี (0–25) ข้างชื่อจุด")
            frame_stride = gr.Slider(1, 8, value=1, step=1, label="ข้ามเฟรม (frame_stride)")

        # ❗ ไม่มีเอาต์พุตอื่นแล้ว (เหลือเพียง out_vid ตัวเดียว)
        def predict_video(video, conf, show_idx, stride, progress=gr.Progress()):
            # ถ้าไม่มี ffmpeg ให้ไม่คืนไฟล์ (หลีกเลี่ยงส่งข้อความผิดชนิดเข้า gr.Video)
            if not _has_ffmpeg():
                return gr.update()

            vpath = _get_video_path(video)
            if not vpath or (not os.path.exists(vpath)):
                return gr.update()

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                return gr.update()

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            base = os.path.splitext(os.path.basename(vpath))[0]
            uid = uuid.uuid4().hex[:8]
            out_stub = os.path.abspath(f"{base}_pred_{uid}")

            # เปิดตัวเขียน H.264 (stdin raw RGB -> ffmpeg)
            ff_proc, ff_stdin, out_path = _ffmpeg_h264_writer(
                out_stub, max(1.0, float(fps) / max(1, int(stride))), W, H
            )

            frame_idx = 0
            iterable = range(total) if total > 0 else range(10**9)
            for _ in progress.tqdm(iterable, desc="วิเคราะห์วิดีโอ (สร้าง MP4/H.264)"):
                ret, frame = cap.read()
                if not ret:
                    break

                if stride > 1 and (frame_idx % stride != 0):
                    frame_idx += 1
                    continue

                plotted_bgr, _rows = infer_frame_bgr(frame, conf, show_idx, MIN_KP_CONF_DEFAULT)

                # ส่งเฟรมเข้า ffmpeg เป็น RGB24
                ff_stdin.write(cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB).tobytes())

                frame_idx += 1
                if total == 0 and frame_idx > 2000:
                    break

            cap.release()
            try:
                ff_stdin.close()
            except Exception:
                pass
            ff_proc.wait()

            # รอให้ไฟล์พร้อมอ่านเล็กน้อย
            for _ in range(20):
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    break
                time.sleep(0.1)

            # คืน "ไฟล์วิดีโอ" เพียงอย่างเดียวตามที่ต้องการ
            return out_path

        run_video_btn = gr.Button("ทำนายทั้งวิดีโอ")
        run_video_btn.click(
            fn=predict_video,
            inputs=[in_vid, conf_v, show_index_v, frame_stride],
            outputs=[out_vid],
            queue=True,
        )

app.queue().launch(share=True, debug=True)

