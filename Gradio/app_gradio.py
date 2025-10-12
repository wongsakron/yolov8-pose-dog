# app.py
import os, csv, time
import uuid
import cv2
import numpy as np
import gradio as gr

from set_modal import (
    infer,            # ใช้กับภาพนิ่ง
    infer_frame_bgr,  # ใช้ประมวลผลทีละเฟรมของวิดีโอ
    MIN_KP_CONF_DEFAULT
)

def _abs(p):
    return os.path.abspath(p) if p else p

# ========== UI: ภาพนิ่ง ==========
with gr.Blocks(title="Dog Pose (Thai Labels) – YOLO + Gradio") as app:
    gr.Markdown("## 🐶 Dog Pose Estimation (Thai Labels)\nอัปโหลดภาพหมาหรือวิดีโอ แล้วระบบจะทำนาย keypoints พร้อมป้ายชื่อภาษาไทย")

    history = gr.State([])

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
            # Gradio v5: ใช้ gr.update แทน gallery.update(...)
            return out_np, gr.update(value=hist), hist

        # ปุ่มทำนายภาพนิ่ง
        run_btn.click(
            fn=predict_and_store,
            inputs=[inp, conf, show_index, history],
            outputs=[out_img, gallery, history],
        )

        # อัปโหลดไฟล์ภาพแล้วรันทันที
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
                label="วิดีโอผลลัพธ์ (ทำนายทั้งคลิป)",
                interactive=False,
                autoplay=True,
                show_download_button=True,
            )

        with gr.Row():
            conf_v = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="ค่าความมั่นใจขั้นต่ำ (conf)")
            show_index_v = gr.Checkbox(value=False, label="โชว์เลขดัชนี (0–25) ข้างชื่อจุด")
            frame_stride = gr.Slider(1, 8, value=1, step=1, label="ข้ามเฟรม (frame_stride)")
            save_csv = gr.Checkbox(value=False, label="บันทึก CSV ค่าจุด (ต่อเฟรม)")

        csv_download = gr.File(label="ดาวน์โหลด CSV (ถ้าเลือกบันทึก)")
        processing = gr.State(value=False)  # กันกดซ้ำ

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

        def predict_video(video, conf, show_idx, stride, want_csv, is_busy, progress=gr.Progress()):
                if is_busy:
                    return gr.update(), None, True
                is_busy = True

                # --- ดึง path ไฟล์จาก Gradio ---
                def _get_video_path(v):
                    if v is None: return None
                    if isinstance(v, str): return v if os.path.exists(v) else None
                    if isinstance(v, dict):
                        for k in ("path", "name", "tempfile", "file"):
                            p = v.get(k)
                            if isinstance(p, str) and os.path.exists(p):
                                return p
                    for attr in ("name", "path"):
                        p = getattr(v, attr, None)
                        if isinstance(p, str) and os.path.exists(p):
                            return p
                    return None

                vpath = _get_video_path(video)
                if not vpath:
                    return gr.update(), None, False

                cap = cv2.VideoCapture(vpath)
                if not cap.isOpened():
                    return gr.update(), None, False

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                base = os.path.splitext(os.path.basename(vpath))[0]
                uid = uuid.uuid4().hex[:8]

                # ✅ เลี่ยง H.264 (ต้องมี openh264) → ใช้ mp4v ก่อน, ไม่ได้ค่อย XVID
                codec_candidates = [
                    ("mp4v", f"{base}_pred_{uid}.mp4"),
                    ("XVID", f"{base}_pred_{uid}.avi"),
                ]

                writer = None
                out_video_path = None
                for fourcc_name, filename in codec_candidates:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    tmp_path = os.path.abspath(filename)
                    w = cv2.VideoWriter(tmp_path, fourcc, max(1.0, fps / max(1, stride)), (W, H))
                    if w.isOpened():
                        writer = w
                        out_video_path = tmp_path
                        break
                if writer is None:
                    cap.release()
                    return gr.update(), None, False

                out_csv_path = os.path.abspath(f"{base}_pred_{uid}.csv") if want_csv else None
                csv_file = None
                csv_writer = None
                if want_csv:
                    csv_file = open(out_csv_path, "w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["frame_idx", "person", "kp_idx", "kp_name_th", "x", "y", "conf"])

                frame_idx = 0
                iterable = range(total) if total > 0 else range(10**9)
                for _ in progress.tqdm(iterable, desc="วิเคราะห์วิดีโอ"):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if stride > 1 and (frame_idx % stride != 0):
                        frame_idx += 1
                        continue

                    plotted_bgr, rows = infer_frame_bgr(frame, conf, show_idx, MIN_KP_CONF_DEFAULT)
                    writer.write(plotted_bgr)

                    if csv_writer and rows:
                        for r in rows:
                            csv_writer.writerow([frame_idx] + r)

                    frame_idx += 1
                    if total == 0 and frame_idx > 2000:
                        break

                cap.release()
                writer.release()
                if csv_file:
                    csv_file.close()

                # รอให้ไฟล์พร้อมอ่านเล็กน้อย
                for _ in range(10):
                    if out_video_path and os.path.exists(out_video_path) and os.path.getsize(out_video_path) > 0:
                        break
                    time.sleep(0.1)

                # ❗ Gradio v5: ส่ง “พาธไฟล์ (string)” ตรง ๆ ให้กับ gr.Video
                return out_video_path, (out_csv_path if (want_csv and os.path.exists(out_csv_path)) else None), False

        run_video_btn = gr.Button("ทำนายทั้งวิดีโอ")
        run_video_btn.click(
            fn=predict_video,
            inputs=[in_vid, conf_v, show_index_v, frame_stride, save_csv, processing],
            outputs=[out_vid, csv_download, processing],
            queue=True,
        )

app.queue().launch()
