import gradio as gr
import yt_dlp
import os
from tools.step010_demucs_vr import separate_all_audio_under_folder
from tools.step020_asr import transcribe_all_audio_under_folder
from tools.step030_translation import translate_all_transcript_under_folder
from tools.step040_tts import generate_all_wavs_under_folder
from tools.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

# Fix: Updated function to handle redirects and use yt-dlp properly
def download_from_url(url, output_folder="videos", resolution="1080p"):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ydl_opts = {
            'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
            'format': f'bestvideo[height<={resolution}]+bestaudio/best',
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"Download completed: {url}"
    except Exception as e:
        return f"Download failed: {e}"

# One-click automation
full_auto_interface = gr.Interface(
    fn=do_everything,
    inputs=[
        gr.Textbox(label='Video output folder', value='videos'),
        gr.Textbox(label='Video URL', placeholder='Enter video URL (YouTube/Bilibili)', value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        gr.Slider(minimum=1, maximum=100, step=1, label='Number of videos', value=5),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p'),
        gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx'], label='Audio Model', value='htdemucs_ft'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='Compute Device', value='auto'),
        gr.Slider(minimum=0, maximum=10, step=1, label='Shifts', value=5),
        gr.Dropdown(['WhisperX', 'FunASR'], label='ASR Model', value='WhisperX'),
        gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='WhisperX Model Size', value='large'),
        gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32),
        gr.Checkbox(label='Separate Speakers', value=True),
        gr.Radio([None, 1, 2, 3, 4, 5], label='Min Speakers', value=None),
        gr.Radio([None, 1, 2, 3, 4, 5], label='Max Speakers', value=None),
        gr.Dropdown(['OpenAI', 'LLM', 'Google Translate'], label='Translation Method', value='LLM'),
        gr.Dropdown(['Chinese', 'English', 'Japanese', 'Korean'], label='Target Language', value='Chinese'),
        gr.Dropdown(['xtts', 'cosyvoice', 'EdgeTTS'], label='TTS Method', value='xtts'),
        gr.Dropdown(SUPPORT_VOICE, value='zh-CN-XiaoxiaoNeural', label='EdgeTTS Voice'),
        gr.Checkbox(label='Add Subtitles', value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Multiplier', value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label='Frame Rate', value=30),
        gr.Audio(label='Background Music', sources=['upload']),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='Music Volume', value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='Video Volume', value=1.0),
    ],
    outputs=[gr.Text(label='Process Status'), gr.Video(label='Generated Video')],
    allow_flagging='never',
)

# Video Download
download_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(label='Video URL', placeholder='Enter video URL (YouTube/Bilibili)', value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        gr.Textbox(label='Video output folder', value='videos'),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p'),
    ],
    outputs=[gr.Textbox(label='Download Status')],
    allow_flagging='never',
)

# Vocal Separation
demucs_interface = gr.Interface(
    fn=separate_all_audio_under_folder,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Text(label='Separation Status')],
    allow_flagging='never',
)

# ASR Speech Recognition
asr_interface = gr.Interface(
    fn=transcribe_all_audio_under_folder,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Text(label='ASR Status')],
    allow_flagging='never',
)

# Translation
translation_interface = gr.Interface(
    fn=translate_all_transcript_under_folder,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Text(label='Translation Status')],
    allow_flagging='never',
)

# Text-to-Speech
tts_interface = gr.Interface(
    fn=generate_all_wavs_under_folder,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Text(label='TTS Status')],
    allow_flagging='never',
)

# Video Synthesis
synthesize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Text(label='Synthesis Status')],
    allow_flagging='never',
)

# Linly-Talker (Under Development)
linly_talker_interface = gr.Interface(
    fn=lambda: None,
    inputs=[gr.Textbox(label='Video Folder', value='videos')],
    outputs=[gr.Markdown("Under development. Stay tuned!")],
)

# Gradio Theme
my_theme = gr.themes.Soft()

# Application Tabs
app = gr.TabbedInterface(
    theme=my_theme,
    interface_list=[
        full_auto_interface,
        download_interface,
        demucs_interface,
        asr_interface,
        translation_interface,
        tts_interface,
        synthesize_video_interface,
        linly_talker_interface
    ],
    tab_names=[
        'One-Click Automation', 'Download Videos', 'Vocal Separation', 
        'Speech Recognition', 'Subtitle Translation', 'Speech Synthesis', 'Video Synthesis', 
        'Linly-Talker (WIP)'
    ],
    title='Smart Video Multi-Language Dubbing - Linly-Dubbing'
)

# Run Application
if __name__ == '__main__':
    app.launch(
        server_name="127.0.0.1", 
        server_port=6006,
        share=True,
        inbrowser=True
    )
