import base64
import io
import json
import os
from datetime import datetime

from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from df.enhance import enhance, init_df
from pyannote.audio import Pipeline
from transformers import pipeline, GenerationConfig

import google.generativeai as genai

from fastapi import FastAPI, UploadFile, File, WebSocket, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from pydantic import BaseModel
from enum import Enum

from .chunkformer_asr import endless_decode, init
from .utils import correct_vietnamese_text, correct_vietnamese_text_gemini

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

try:
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
except Exception as e:
    print("⚠️  Pyannote initialisation failed:", e)
    diar_pipeline = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def load_audio_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """
    Pydub‑based loader that mimics the Chunkformer `load_audio` function
    but works on raw bytes instead of a file path.
    
    Returns:
        wav (torch.Tensor) – shape [1, T] ‑ 32‑bit float, *not* normalised
        sr  (int)          – always 16 000 Hz
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(data))
        audio = (audio.set_frame_rate(16_000)
                    .set_sample_width(2)      # 16‑bit PCM
                    .set_channels(1))         # mono
        wav = torch.as_tensor(audio.get_array_of_samples(),
                            dtype=torch.float32).unsqueeze(0)
        return wav, 16_000
    except CouldntDecodeError:
        if len(data) % 2 == 0:                      # divisible by 2 → try int16
            pcm = np.frombuffer(data, dtype=np.int16)
            wav = torch.from_numpy(pcm.astype(np.float32))     # cast keeps range
        else:                                       # otherwise fall back to f32
            pcm = np.frombuffer(data, dtype=np.float32)
            wav = torch.from_numpy(pcm * 32768.0)    # re‑scale −1…1 → int16 range

        wav = wav.unsqueeze(0)                      # [1, T] like load_audio()
        return wav, 16_000

# from post_processing import init as pp_init

# # Load resources once (paths can come from env vars or hard-code)
# DICT_PATH  = "/home/tuquan/api/asr/vi_dictionary.txt"
# KENLM_PATH = "/home/tuquan/api/asr/vi_lm_4grams.bin"

# (
#     kenlm_model,
#     symspell_telex,
#     bert_model,            # we keep the variable name; may be None
#     tokenizer,
#     reverse_telex_map,
# ) = pp_init(
#     DICT_PATH,
#     KENLM_PATH,
#     use_bert=False,
#     bert_model_name="FacebookAI/xlm-roberta-large",
#     sym_max_dict_edit_distance=3,
#     dictionary_json_key="text",
# )

CHUNKFORMER_PATH = "/home/tuquan/api/asr/chunkformer-large-vie"
chunkformer_model, char_dict = init(CHUNKFORMER_PATH, torch.device('cuda'))

def maybe_enhance_text(text: str, use_enhance: bool) -> str:
    if not use_enhance:
        return text
    return correct_vietnamese_text_gemini(text)

df_model, df_state, _ = init_df()

def maybe_denoise(wav, use_denoise: bool) -> torch.Tensor:
    """
    If use_denoise=True, run DeepFilterNet; else return the tensor unchanged.
    Expects 1-D or [1, T] mono tensor at df_state.sr().
    """
    if not use_denoise:
        return wav
    enhanced = enhance(df_model, df_state, wav)
    if enhanced.dim() == 1:
        enhanced = enhanced.unsqueeze(0)
    return enhanced 

def tensor_to_b64wav(wav_tensor: torch.Tensor, sr: int) -> str:
    """Return base64-encoded WAV bytes (mono)."""
    buf = io.BytesIO()
    sf.write(buf, wav_tensor.squeeze(0).cpu().numpy(), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device="cuda:1")
pipe.model.generation_config = GenerationConfig.from_pretrained(
    pipe.model.name_or_path,
    suppress_tokens=[], begin_suppress_tokens=[], force_words_ids=None,
)
generate_kwargs = {"language": "vi"}

@app.post("/api/transcribe")
async def transcribe_audio(
    file    : UploadFile = File(...),
    denoise : bool       = Form(False),
    diarize : bool       = Form(False),
    enhance : bool       = Form(False),
):
    try:
        orig_b64, enh_b64 = None, None
        data = await file.read()
        # optional denoise
        if denoise:
            wav, sr = torchaudio.load(io.BytesIO(data))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != df_state.sr():
                wav = torchaudio.transforms.Resample(sr, df_state.sr())(wav)
                sr  = df_state.sr()
            raw_wav = wav.clone()
            wav = maybe_denoise(wav, denoise) 
            orig_b64 = tensor_to_b64wav(raw_wav, sr)          # original @48 kHz (or 16)
            # wav has already been denoised by maybe_denoise
            enh_b64  = tensor_to_b64wav(wav, sr)
        # load & mono-ise
        wav, sr = load_audio_bytes(data)     # [C,T]

        # diarisation path
        if diarize and diar_pipeline:
            # 1) ensure 16-kHz mono torch tensor, shape [1, T]
            wav16 = torchaudio.transforms.Resample(sr, 16_000)(wav).contiguous()
            # (wav is [1,T] already, so squeeze is NOT needed)

            diar_result = diar_pipeline({
                "waveform": wav16,              # ← torch.tensor, not numpy
                "sample_rate": 16_000
            })

            segments = []
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
                beg = int(turn.start * 16_000)
                end = int(turn.end   * 16_000)

                snippet = wav16[:, beg:end]     # still torch, [1, N]

                # text = endless_decode(snippet, chunkformer_model, char_dict)
                text = pipe(snippet.squeeze(0).numpy(),
                            return_timestamps=True,
                            generate_kwargs=generate_kwargs)["text"]
                text = maybe_enhance_text(text, enhance)

                segments.append({
                    "speaker": speaker,
                    "start"  : int(turn.start * 1000),
                    "end"    : int(turn.end   * 1000),
                    "text"   : text.strip()
                })

            return {
                "segments": segments,
                "original_audio":  orig_b64,
                "enhanced_audio":  enh_b64
            }

        # txt = endless_decode(wav, chunkformer_model, char_dict)
        txt = pipe(wav.squeeze(0).numpy(),
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs)["text"]
        
        txt = maybe_enhance_text(txt, enhance)

        payload = {"transcription": txt}
        if orig_b64:
            payload["original_audio"]  = orig_b64
            payload["enhanced_audio"]  = enh_b64
        return payload

    # error handling
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

class Lang(str, Enum):
    vi = "vi"   # Vietnamese
    en = "en"   # English
    zh = "zh"   # Chinese
    fr = "fr"   # French

class SummarizeRequest(BaseModel):
    text: str
    lang: Lang = Lang.vi
    
@ app.post("/api/summarize_text")
async def summarize_text(req: SummarizeRequest):
    """
    Summarise `req.text` in the language specified by `req.lang`.

    The reply is returned as three bullet-points:
        1. Main purpose / gist
        2. Key points
        3. Conclusions / recommendations
    """
    PROMPTS = {
        "vi": (
            "Bạn là chuyên gia xử lý tài liệu. Hãy tóm tắt đoạn văn sau thành **ba mục**:\n"
            "1. Mục đích hoặc nội dung chính\n"
            "2. Các điểm quan trọng\n"
            "3. Kết luận / đề xuất (nếu có)\n\n"
            "Văn bản:\n\"\"\"{text}\"\"\""
        ),
        "en": (
            "You are a document-processing expert. Summarise the passage below in **three bullets**:\n"
            "1. Main purpose / gist\n"
            "2. Key points\n"
            "3. Conclusions / recommendations (if any)\n\n"
            "Text:\n\"\"\"{text}\"\"\""
        ),
        "zh": (
            "你是一位文档处理专家。请将下段文字用**三条要点**摘要：\n"
            "1. 主要目的 / 核心内容\n"
            "2. 关键要点\n"
            "3. 结论 / 建议（如有）\n\n"
            "正文：\n\"\"\"{text}\"\"\""
        ),
        "fr": (
            "Vous êtes expert en traitement de documents. Résumez le texte ci-dessous en **trois points** :\n"
            "1. Objet principal / idée générale\n"
            "2. Points clés\n"
            "3. Conclusions / recommandations (le cas échéant)\n\n"
            "Texte :\n\"\"\"{text}\"\"\""
        ),
    }

    prompt = PROMPTS[req.lang.value].format(text=req.text)

    try:
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        # Gemini may return nested errors; stringify for the client
        return JSONResponse(status_code=500, content={"error": str(e)})

@ app.websocket("/transcribe")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            payload  = json.loads(await ws.receive_text())
            data = base64.b64decode(payload["audio"])
            if not data:
                continue
            wav, _ = load_audio_bytes(data)                     # [1,T] @16k

            # --- optional DeepFilterNet denoise --------------------------------
            if payload.get("denoise", False):
                wav48 = torchaudio.transforms.Resample(16_000, df_state.sr())(wav)
                wav48 = maybe_denoise(wav48, True)
                wav   = torchaudio.transforms.Resample(df_state.sr(), 16_000)(wav48)

            # --- Whisper ASR (no enhancement here!) ---------------------------
            text = endless_decode(wav, chunkformer_model, char_dict)

            await ws.send_json({"status": "success", "transcription": text})

    except WebSocketDisconnect:
        print("WebSocket disconnected")

from datetime import datetime
import os

RECORD_DIR = "recordings"
os.makedirs(RECORD_DIR, exist_ok=True)

@app.post("/api/upload_recording")
async def upload_recording(file: UploadFile = File(...)):
    """
    Receive a complete recording (WAV/webm/ogg) and save it under
    recordings/YYYYmmdd_HHMMSS_<original-name>.ext
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(RECORD_DIR, f"{ts}_{file.filename}")
    with open(dst, "wb") as fh:
        fh.write(await file.read())
    return {"saved_as": dst}

class EnhanceRequest(BaseModel):
    text: str
    enhance: bool = True                 # keep API flexible for later

@ app.post("/api/enhance_text")
async def enhance_text(req: EnhanceRequest):
    try:
        improved = maybe_enhance_text(req.text, req.enhance)
        return {"text": improved}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
