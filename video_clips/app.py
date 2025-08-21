from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
import uvicorn
import os
import subprocess
import shutil
import cv2
import numpy as np
import yt_dlp
import base64
import requests
import json
import re
import zipfile
import mediapipe as mp
from constants import GEMINI_API_KEY, GEMINI_API_URL


my_prompt = '''You are a transcription assistant. I will provide an audio file. Please transcribe it and return the transcript as an array of segments. 

Requirements for each segment:
-Each segment should correspond to a single sentence.
-Include the estimated start and end time (in seconds) for each segment.
-Try to estimate the timing as accurately as possible, even if approximate.
-The output must be in valid JSON format, like this example:
[
  {
    "start": 0.0,
    "end": 3.5,
    "text": "Hello, how are you?"
  },
  {
    "start": 3.5,
    "end": 7.0,
    "text": "The weather is very nice."
  }
]
'''


app = FastAPI()
CANDIDATE_LABELS = ["mutlu", "üzücü", "öfkeli", "korkunç", "şaşırtıcı", "eğlenceli", "bilgilendirici", "heyecanlı"]

 

def download_youtube_video(url, output_dir = "temp_outputs"):
    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, "input.mp4")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print("Video indirme hatası:", e)
        return None
    
    return output_path


def extract_audio(video_path, audio_path):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path])

def transcript(audio_path, segment_duration = 10):
    global GEMINI_API_KEY
    global my_prompt

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")


    payload = {
        "contents": [
            {
                "parts": [
                    {"text": my_prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_base64
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    response = requests.post(
        GEMINI_API_URL,
        headers=headers, json=payload
    )
    print(response.status_code)
    print(response.text) 
    
    data = response.json()
    transcript_segments = []

    try:
        transcript_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # ```json ... ``` kısmını temizle
        transcript_text = re.sub(r"^```json\s*|\s*```$", "", transcript_text, flags=re.DOTALL)

        matches = re.findall(
            r'\{\s*"start"\s*:\s*([\d.]+),\s*"end"\s*:\s*([\d.]+),\s*"text"\s*:\s*"([^"]*)"\s*\}',
            transcript_text
        )

        # Liste haline getir
        transcript_segments = [
            {"start": float(s), "end": float(e), "text": t} 
            for s, e, t in matches
        ]

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print("Transcript parsing error:", e)
        transcript_segments = []

    return transcript_segments
    
    

def emotion_score(text: str, emotion: str) -> float:

    global GEMINI_API_KEY, CANDIDATE_LABELS

    my_prompt_score = f"""
    Sen cümleleri aşağıdaki duygulara göre labellayan bir asistansın.
    Her label için cümlenin o duyguyu ne kadar ifade ettiğini 0-1 arası bir skor ile değerlendir.
    Sadece JSON çıktısı döndür, ekstra açıklama yazma.
    JSON formatı: [["label", score], ...]
    Cümle: "{text}"
    Candidate labels: {CANDIDATE_LABELS}
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": my_prompt_score}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers=headers, json=payload
    )

    if response.status_code != 200:
        print("Gemini API error:", response.status_code, response.text)
        return 0.0

    try:
        data = response.json()
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]

        response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.DOTALL)

        scores_list = json.loads(response_text)

        target_score = 0.0
        for label, score in scores_list:
            if label.lower() == emotion.lower():
                if score > 0.6:
                    score = 1
                else:
                    score /= 2
                target_score = float(score)
                break

        return target_score

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print("Emotion parsing error:", e)
        return 0.0


def compute_total_scores(video_path, transcript_segments, emotion):

    bert_scores = []
    for seg in transcript_segments:
        score = emotion_score(seg['text'], emotion)
        bert_scores.append({'start': seg['start'], 'end': seg['end'], 'score': score})

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    total_scores = []

    for i in range(frame_count):
        t = i / fps
        
        b_score = 0
        for seg in bert_scores:
            if seg['start'] <= t <= seg['end']:
                b_score = seg['score']
        total_scores.append(b_score)

    return np.array(total_scores), fps

def get_highlights(total_scores, fps, highlight_duration, number_of_videos):
    scores_copy = total_scores.copy()
    highlights = []
    window = int(highlight_duration * fps)

    for _ in range(number_of_videos):
        if len(scores_copy) < window:
            break
    
        max_sum = 0
        max_start = 0

        for i in range(len(scores_copy) - window):
            s = sum(scores_copy[i:i + window])
            if s > max_sum:
                max_sum = s
                max_start = i

        start_time = max_start / fps
        end_time = start_time + highlight_duration
        highlights.append((start_time, end_time))

        scores_copy[max_start:max_start+window] = 0

    return highlights

def cut_video(video_path, start_time, end_time, output_path):

    duration = end_time - start_time

    subprocess.call(['ffmpeg', '-y', 
                     '-ss', str(start_time), 
                     '-i', video_path, 
                     '-t', str(duration),
                     '-c:v', 'libx264', 
                     '-c:a', 'aac', 
                     output_path
                    ])
    

def make_vertical(input_path, output_path, target_width=1080, target_height=1920, alpha=0.2):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (target_width, target_height))
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5)
    ema_center_x = None
    ema_center_y = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            x_coords = [lm[i].x for i in range(len(lm))]
            y_coords = [lm[i].y for i in range(len(lm))]
            center_x = int(np.mean(x_coords) * frame.shape[1])
            center_y = int(np.mean(y_coords) * frame.shape[0])
        else:
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
        if ema_center_x is None:
            ema_center_x = center_x
            ema_center_y = center_y
        else:
            ema_center_x = int(alpha * center_x +
                               (1 - alpha) * ema_center_x)
            ema_center_y = int(alpha * center_y +
                               (1 - alpha) * ema_center_y)
        start_x = max(0, ema_center_x - target_width // 2)
        start_y = max(0, ema_center_y - target_height // 2)
        if start_x + target_width > frame.shape[1]:
            start_x = frame.shape[1] - target_width
        if start_y + target_height > frame.shape[0]:
            start_y = frame.shape[0] - target_height
        crop_frame = frame[start_y:start_y+target_height,
                           start_x:start_x+target_width]
        crop_frame = cv2.resize(crop_frame, (target_width, target_height))
        out.write(crop_frame)
    cap.release()
    out.release()
    


@app.post("/requirements")
async def require(states: dict = Body(..., embed=True)):

    seconds = states["duration"]
    emotion = states["emotion"]
    number_of_videos = states["number_of_videos"]
    video_link = states["video_link"]

    
    if os.path.exists("temp_outputs"):
        shutil.rmtree("temp_outputs")
    os.makedirs("temp_outputs", exist_ok = True)

    video_path = download_youtube_video(video_link)
    audio_path = "temp_outputs/audio.wav"
    extract_audio(video_path, audio_path)
    transcript_segments = transcript(audio_path)

    total_scores, fps = compute_total_scores(video_path, transcript_segments, emotion)
    highlights = get_highlights(total_scores, fps, seconds, number_of_videos)

    output_dir = "temp_outputs/results" 
    os.makedirs(output_dir, exist_ok=True) 

    output_files = []
    
    for idx, (start_time, end_time) in enumerate(highlights):
        raw_output = os.path.join(output_dir, f"raw_{idx}.mp4")
        final_output = os.path.join(output_dir, f"result_{idx}.mp4")
        cut_video(video_path, start_time, end_time, raw_output)
        make_vertical(raw_output, final_output)

        if os.path.exists(final_output):
            output_files.append(final_output)


    zip_path = "temp_outputs/results.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, file in enumerate(output_files):
            zipf.write(file, f"result_{idx}.mp4")


    return {"segments": transcript_segments}


ZIP_PATH = "temp_outputs/results.zip" 

@app.get("/download_results")
async def download_results():
    if os.path.exists(ZIP_PATH):
        return FileResponse(ZIP_PATH, media_type="application/zip", filename="results.zip")
    else:
        return {"error": "Zip dosyası bulunamadı. Önce /requirements endpoint'ini çalıştırın."}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)