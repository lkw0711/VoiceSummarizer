import gradio as gr
import os
import shutil
import tempfile
from openai import OpenAI

tmp_dir = "./tmp"
audio_path = os.path.join(tmp_dir, audio_file.name)
shutil.copyfile(audio_file.name, audio_path)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

client = OpenAI(
    api_key = "sess-YiinApevNVx19Audctze9DzuE3yLe04PvtcNGSf5"
)


def transcribe_audio(audio_file):
    with open(audio_file.name, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript['data']['text']


def extract_key_points(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        prompt=f"Summarize the key points: {text}",
        max_tokens=150
    )
    return response['choices'][0]['text']


def voicesummarizer(audio_file):
    # 將上傳的音頻文件保存到臨時位置
    audio_path = gr.save_file(audio_file, "./audio")

    # 調用 transcribe_audio 和 extract_key_points
    # transcribed_text = transcribe_audio(audio_path)
    # key_points = extract_key_points(transcribed_text)

    # return transcribed_text, key_points


# 創建 gradio 界面
app = gr.Interface(
    fn=voicesummarizer,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", format="mp3",
                    label="Upload Audio File"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Key Points")
    ],
    title="VoiceSummarizer - Audio to Summary",
    description="Upload your audio file and get a transcribed text along with its key points."
)

if __name__ == "__main__":
    app.launch()
# share=True


