import gradio as gr
from set_modal import infer  # infer คืน (out_img, out_tbl)

with gr.Blocks(title="Dog Pose (Thai Labels) – YOLO + Gradio") as app:
    gr.Markdown("## 🐶 Dog Pose Estimation (Thai Labels)\nอัปโหลดภาพหมา แล้วระบบจะทำนาย keypoints พร้อมป้ายชื่อภาษาไทย")

    history = gr.State([])

    with gr.Row():
        # ไม่ต้องล้างค่าอัตโนมัติ และให้ upload ทับได้ทันที
        inp = gr.Image(type="pil", label="อัปโหลดภาพ", container=True)
        out_img = gr.Image(type="numpy", label="ผลลัพธ์", interactive=False)

    with gr.Row():
        conf = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="ค่าความมั่นใจขั้นต่ำ (conf)")
        show_index = gr.Checkbox(value=False, label="โชว์เลขดัชนี (0–25) ข้างชื่อจุด")

    run_btn = gr.Button("ทำนาย")

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
        out_img, _ = infer(image, conf, show_index)
        if out_img is not None:
            hist = (hist or []) + [out_img]
            if len(hist) > 30:
                hist = hist[-30:]
        return out_img, hist, hist

    # 1) กดปุ่มทำนาย
    run_btn.click(
        fn=predict_and_store,
        inputs=[inp, conf, show_index, history],
        outputs=[out_img, gallery, history],
    )

    # 2) อัปโหลดรูปใหม่ (รวมถึงไฟล์เดิมซ้ำ) → ใช้ .upload() แทน .change()
    inp.upload(
        fn=predict_and_store,
        inputs=[inp, conf, show_index, history],
        outputs=[out_img, gallery, history],
    )

app.queue().launch()
