from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
import uvicorn
import os
import subprocess
import shutil
import cv2
import librosa
import numpy as np
import yt_dlp
import base64
import requests
import json
import base64
import re
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
emotion = ''
CANDIDATE_LABELS = ["mutlu", "üzücü", "öfkeli", "korkunç", "şaşırtıcı", "eğlenceli", "bilgilendirici", "heyecanlı"]

 

def download_youtube_video(url, output_dir = "temp_outputs"):
    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, "input.mp4")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
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


def opencv_motion_score(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        cap.release()
        return np.array([])
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        scores.append(diff.sum())
        prev_gray = gray
    cap.release()
    
    scores = np.array(scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores

def librosa_audio_score(audio_path):
    y, sr = librosa.load(audio_path, sr = None, mono = True)
    rms = librosa.feature.rms(y=y)[0]
    if rms.max() > 0:
        rms = rms / rms.max()
    return rms

def compute_total_scores(video_path, audio_path, transcript_segments, emotion):
    motion_scores = opencv_motion_score(video_path)
    audio_scores = librosa_audio_score(audio_path)

    
    bert_scores = []
    for seg in transcript_segments:
        score = emotion_score(seg['text'], emotion)
        bert_scores.append({'start': seg['start'], 'end': seg['end'], 'score': score})

    
    frame_count = len(motion_scores)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_scores = []

    for i in range(frame_count):
        t = i / fps
        
        b_score = 0
        for seg in bert_scores:
            if seg['start'] <= t <= seg['end']:
                b_score = seg['score']
        total = motion_scores[i] + audio_scores[min(i, len(audio_scores)-1)] + b_score
        total_scores.append(total)

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
    


@app.post("/requirements")
async def require(states: dict = Body(..., embed=True)):

    seconds = states["duration"]
    emotion = states["emotion"]
    number_of_videos = states["number_of_videos"]
    video_link = states["video_link"]

    print({"seconds": seconds, "emotion": emotion, "video_link": video_link})
    
    if os.path.exists("temp_outputs"):
        shutil.rmtree("temp_outputs")
    os.makedirs("temp_outputs", exist_ok = True)

    video_path = download_youtube_video(video_link)
    audio_path = "temp_outputs/audio.wav"
    extract_audio(video_path, audio_path)
    transcript_segments = transcript(audio_path)

    total_scores, fps = compute_total_scores(video_path, audio_path, transcript_segments, emotion)
    highlights = get_highlights(total_scores, fps, seconds, number_of_videos)

    output_dir = "temp_outputs/results" 
    os.makedirs(output_dir, exist_ok=True) 

    output_files = []
    for idx, (start_time, end_time) in enumerate(highlights):
        output_path = os.path.join(output_dir, f"result_{idx}.mp4")
        cut_video(video_path, start_time, end_time, output_path)
        if os.path.exists(output_path):
            output_files.append(output_path)


    zip_path = "temp_outputs/results.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, file in enumerate(output_files):
            zipf.write(file, f"result_{idx}.mp4")


    return {"segments": transcript_segments}



from fastapi.responses import FileResponse

ZIP_PATH = "temp_outputs/results.zip" 

@app.get("/download_results")
async def download_results():
    if os.path.exists(ZIP_PATH):
        return FileResponse(ZIP_PATH, media_type="application/zip", filename="results.zip")
    else:
        return {"error": "Zip dosyası bulunamadı. Önce /requirements endpoint'ini çalıştırın."}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)