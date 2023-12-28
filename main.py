import gradio as gr
import os
import tempfile
from openai import OpenAI


tempfile.tempdir = ".\\tmp"
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

input_audio = gr.Audio(sources=["upload", "microphone"],
                       type="filepath", format="mp3", label="Upload Audio File")
input_language = gr.Dropdown(
    choices=[["中文", "zh"], ["English", "en"]], label="Audio_language")
key_points_prompt = gr.Textbox(value="",
                               placeholder="to summarize the key points")
output_language = gr.Dropdown(
    choices=[["中文", "繁體中文"], ["English", "English"]], label="key_points_language")
client = OpenAI(
    api_key="sess-YiinApevNVx19Audctze9DzuE3yLe04PvtcNGSf5"
)


def transcribe_audio(audio_file, language):
    with open(audio_file, 'rb') as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="text"
        )
    return transcript


def extract_key_points(text, language, prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system",
                "content": f"You will be provided with an audio transcription, and your task is to summarize the key points, and {prompt}, with {language}"},
            {"role": "user", "content": f"{text}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content


def voicesummarizer(audio_file, input_language, key_points_prompt, output_language):
    transcribed_text = transcribe_audio(audio_file, input_language)
    key_points = extract_key_points(
        transcribed_text, output_language, key_points_prompt)
    return transcribed_text, key_points


app = gr.Interface(
    fn=voicesummarizer,
    inputs=[input_audio, input_language, key_points_prompt, output_language],
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Key Points")
    ],
    title="VoiceSummarizer - Audio to Summary",
    description="Upload your audio file and get a transcribed text along with its key points.",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch()
    # share=True)
