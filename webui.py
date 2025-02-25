import gradio as gr
from tools.step000_video_downloader import download_from_url
from tools.step010_demucs_vr import separate_all_audio_under_folder
from tools.step020_asr import transcribe_all_audio_under_folder
from tools.step030_translation import translate_all_transcript_under_folder
from tools.step040_tts import generate_all_wavs_under_folder
from tools.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

# One-click automation interface
full_auto_interface = gr.Interface(
    fn=do_everything,
    inputs=[
        gr.Textbox(label='Video output folder', value='videos'),
        gr.Textbox(label='Video URL', placeholder='Please enter the URL of the video, playlist or channel on Youtube or Bilibili', 
                   value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        gr.Slider(minimum=1, maximum=100, step=1, label='Number of downloaded videos', value=5),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='resolution', value='1080p'),

        gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label='Model', value='htdemucs_ft'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='computing equipment', value='auto'),
        gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5),

        gr.Dropdown(['WhisperX', 'FunASR'], label='ASR model selection', value='WhisperX'),
        gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='WhisperX model size', value='large'),
        gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32),
        gr.Checkbox(label='Separate multiple speakers', value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Minimum number of speakers', value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Maximum number of speakers', value=None),

        gr.Dropdown(['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie'], label='Translation method', value='LLM'),
        gr.Dropdown(['Simplified Chinese', 'Traditional Chinese', 'English', 'Cantonese', 'Japanese', 'Korean'], label='target language', value='Simplified Chinese'),

        gr.Dropdown(['xtts', 'cosyvoice', 'EdgeTTS'], label='AI speech generation method', value='xtts'),
        gr.Dropdown(['Chinese', 'English', 'Cantonese', 'Japanese', 'Korean', 'Spanish', 'French'], label='target language', value='Chinese'),
        gr.Dropdown(SUPPORT_VOICE, value='zh-CN-XiaoxiaoNeural', label='EdgeTTS sound selection'),

        gr.Checkbox(label='Add subtitles', value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Acceleration multiple', value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label='Frame rate', value=30),
        gr.Audio(label='background music', sources=['upload']),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='background music volume', value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='video volume', value=1.0),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='resolution', value='1080p'),

        gr.Slider(minimum=1, maximum=100, step=1, label='Max Workers', value=1),
        gr.Slider(minimum=1, maximum=10, step=1, label='Max Retries', value=3),
    ],
    outputs=[gr.Text(label='synthetic state'), gr.Video(label='Synthetic video sample results')],
    allow_flagging='never',
)    

# Download video interface
download_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(label='Video URL', placeholder='Please enter the URL of the video, playlist or channel on Youtube or Bilibili', 
                   value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        gr.Textbox(label='Video output folder', value='videos'),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='resolution', value='1080p'),
        gr.Slider(minimum=1, maximum=100, step=1, label='Number of downloaded videos', value=5),
        # gr.Checkbox(label='single video', value=False),
    ],
    outputs=[
        gr.Textbox(label='Download status'), 
        gr.Video(label='Sample video'), 
        gr.Json(label='Download information')
    ],
    allow_flagging='never',
)

# Vocal separation interface
demucs_interface = gr.Interface(
    fn=separate_all_audio_under_folder,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label='Model', value='htdemucs_ft'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='computing equipment', value='auto'),
        gr.Checkbox(label='Show progress bar', value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5),
    ],
    outputs=[
        gr.Text(label='Separate result status'), 
        gr.Audio(label='Vocal audio'), 
        gr.Audio(label='Accompaniment audio')
    ],
    allow_flagging='never',
)

# AI intelligent speech recognition interface
asr_inference = gr.Interface(
    fn=transcribe_all_audio_under_folder,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Dropdown(['WhisperX', 'FunASR'], label='ASR model selection', value='WhisperX'),
        gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='WhisperX model size', value='large'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='computing equipment', value='auto'),
        gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32),
        gr.Checkbox(label='Separate multiple speakers', value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Minimum number of speakers', value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Maximum number of speakers', value=None),
    ],
    outputs=[
        gr.Text(label='Speech recognition status'), 
        gr.Json(label='Recognition result details')
    ],
    allow_flagging='never',
)

# Translation subtitle interface
translation_interface = gr.Interface(
    fn=translate_all_transcript_under_folder,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Dropdown(['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie'], label='Translation method', value='LLM'),
        gr.Dropdown(['Simplified Chinese', 'Traditional Chinese', 'English', 'Cantonese', 'Japanese', 'Korean'], label='target language', value='Simplified Chinese'),
    ],
    outputs=[
        gr.Text(label='Translation Status'),
        gr.Json(label='Summary results'),
        gr.Json(label='Translation result')
    ],
    allow_flagging='never',
)

# AI speech synthesis interface
tts_interface = gr.Interface(
    fn=generate_all_wavs_under_folder,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Dropdown(['xtts', 'cosyvoice', 'EdgeTTS'], label='AI voice generation method', value='xtts'),
        gr.Dropdown(['Chinese', 'English', 'Cantonese', 'Japanese', 'Korean', 'Spanish', 'French'], label='target language', value='Chinese'),
        gr.Dropdown(SUPPORT_VOICE, value='zh-CN-XiaoxiaoNeural', label='EdgeTTS voice selection'),
    ],
    outputs=[
        gr.Text(label='synthesis status'),
        gr.Audio(label='synthetic speech'),
        gr.Audio(label='original audio')
    ],
    allow_flagging='never',
)

# Video synthesis interface
synthesize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Checkbox(label='Add subtitles', value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Acceleration multiple', value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label='frame rate', value=30),
        gr.Audio(label='background music', sources=['upload'], type='filepath'),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='Background music volume', value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label='Video volume', value=1.0),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='分辨率', value='1080p'),

    ],
    outputs=[
        gr.Text(label='Synthetic status'),
        gr.Video(label='Synthetic video')
    ],
    allow_flagging='never',
)

linly_talker_interface = gr.Interface(
    fn=lambda: None,
    inputs=[
        gr.Textbox(label='video folder', value='videos'),
        gr.Dropdown(['Wav2Lip', 'Wav2Lipv2','SadTalker'], label='AI dubbing method', value='Wav2Lip'),
    ],      
    outputs=[
        gr.Markdown(value="Under construction, please wait for good news. Please refer to [https://github.com/Kedreamix/Linly-Talker](https://github.com/Kedreamix/Linly-Talker)"),
        gr.Text(label='Synthetic state'),
        gr.Video(label='Synthetic video')
    ],
)

my_theme = gr.themes.Soft()

# Application interface
app = gr.TabbedInterface(
    theme=my_theme,
    interface_list=[
        full_auto_interface,
        download_interface,
        demucs_interface,
        asr_inference,
        translation_interface,
        tts_interface,
        synthesize_video_interface,
        linly_talker_interface
    ],
    tab_names=[
        'One-click automation One-Click',
        'Automatically download videos', 'Voice separation', 'AI intelligent speech recognition', 'Subtitle translation', 'AI speech synthesis', 'Video synthesis',
        'Linly-Talker lip sync (under development)'],
    title='Smart video multi-language AI dubbing/translation tool - Linly-Dubbing'
)

if __name__ == '__main__':
    app.launch(
        server_name="127.0.0.1", 
        server_port=6006,
        share=True,
        inbrowser=True
    )
