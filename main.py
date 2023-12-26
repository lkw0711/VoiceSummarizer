import gradio as gr
import os
import shutil
import tempfile
from openai import OpenAI


tempfile.tempdir = ".\\tmp"
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

input_audio = gr.Audio(sources=["upload", "microphone"],
                       type="filepath", format="mp3", label="Upload Audio File")

client = OpenAI(
    api_key="sess-YiinApevNVx19Audctze9DzuE3yLe04PvtcNGSf5"
)


def transcribe_audio(audio_file):
    with open(audio_file, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript


def extract_key_points(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You will be provided with an audio transcript, and your task is to summarize the key points"},
            {"role": "user", "content":  f"{text}"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']


def voicesummarizer(audio_file):
    # 調用 transcribe_audio 和 extract_key_points
    transcribed_text = transcribe_audio(audio_file)
    key_points = extract_key_points(transcribed_text)

    return transcribed_text, key_points


# 創建 gradio 界面
app = gr.Interface(
    fn=voicesummarizer,
    inputs=input_audio,
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
