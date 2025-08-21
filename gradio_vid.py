import gradio as gr
import requests
import shutil
import os

# FastAPI backend URL (FastAPI çalışıyor olmalı)
API_URL = "http://localhost:8000"

def process_video(duration, emotion, number_of_videos, video_link):
    payload = {
        "duration": duration,
        "emotion": emotion,
        "number_of_videos": number_of_videos,
        "video_link": video_link
    }

    # FastAPI /requirements endpoint'ine POST isteği
    response = requests.post(f"{API_URL}/requirements", json=payload)
    
    if response.status_code != 200:
        return f"Hata oluştu: {response.text}", None, None
    
    data = response.json()
    transcript_segments = data.get("segments", [])

    # /download_results endpoint'inden zip dosyasını indir
    download_response = requests.get(f"{API_URL}/download_results", stream=True)
    
    zip_path = None
    if download_response.status_code == 200:
        zip_path = "results.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(download_response.raw, f)

    return "Video işleme tamamlandı.", zip_path, transcript_segments


# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# YouTube Highlight Video Generator")
    
    with gr.Row():
        duration = gr.Number(label="Highlight Süresi (saniye)", value=15)
        emotion = gr.Dropdown(label="Duygu", choices=[
            "mutlu", "üzücü", "öfkeli", "korkunç", 
            "şaşırtıcı", "eğlenceli", "bilgilendirici", "heyecanlı"
        ], value="heyecanlı")
    
    number_of_videos = gr.Number(label="Kaç highlight videosu?", value=2, precision=0)
    video_link = gr.Textbox(label="YouTube Video Linki", placeholder="https://youtu.be/...")
    
    output_text = gr.Textbox(label="Durum")
    output_file = gr.File(label="İndirilen Zip Dosyası")
    output_segments = gr.JSON(label="Transkript Segmentleri")  # Transkript JSON olarak
    
    submit_btn = gr.Button("Video İşle")
    
    submit_btn.click(
        fn=process_video, 
        inputs=[duration, emotion, number_of_videos, video_link], 
        outputs=[output_text, output_file, output_segments]
    )

demo.launch()
