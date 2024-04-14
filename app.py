import gradio as gr
from moodify.moodify import Moodify

if __name__ == "__main__":
    app = Moodify()
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Moodify #")
        with gr.Row():
            with gr.Column():
                img = gr.Image(source="webcam", streaming=True)
                btn = gr.Button(value="Capture")
            with gr.Column():
                output_html = gr.HTML("")
                emotion = gr.Label("")
        btn.click(app.RunMoodify, inputs=img, outputs=[output_html, emotion])

    demo.launch(debug=True)
