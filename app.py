"""
app.py — VibeSync Streamlit Frontend
=====================================

UI Flow:
  1. User uploads a .wav/.mp3 audio file  (or record via browser mic)
  2. Whisper transcribes the audio → raw text
  3. mock_tone_analysis() detects the emotional tone
  4. rephrase_with_vibe() calls the LCEL pipeline → styled output
  5. Results are displayed in a premium card layout
"""

import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VibeSync — Express Your True Tone",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inline CSS — dark glassmorphism theme ─────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Hero header ── */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin: 1.2rem 0;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: box-shadow 0.3s ease;
    }
    .glass-card:hover {
        box-shadow: 0 12px 48px rgba(167,139,250,0.2);
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7c3aed;
        margin-bottom: 0.5rem;
    }

    /* ── Tone badge ── */
    .tone-badge {
        display: inline-block;
        padding: 0.35rem 1.1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .tone-angry   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }
    .tone-happy   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
    .tone-neutral { background: rgba(148,163,184,0.15);color: #94a3b8; border: 1px solid #94a3b8; }
    .tone-sad     { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid #60a5fa; }
    .tone-excited { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }

    /* ── Output box ── */
    .output-box {
        background: linear-gradient(135deg, rgba(167,139,250,0.08), rgba(96,165,250,0.08));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        font-size: 1.2rem;
        line-height: 1.7;
        color: #f1f5f9;
        word-wrap: break-word;
        white-space: pre-wrap;
    }

    /* ── Transcript box ── */
    .transcript-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #7c3aed;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.4rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        font-style: italic;
        line-height: 1.6;
    }

    /* ── Pulse animation for processing ── */
    @keyframes pulse-glow {
        0%   { box-shadow: 0 0 0 0 rgba(167,139,250,0.4); }
        70%  { box-shadow: 0 0 0 14px rgba(167,139,250,0); }
        100% { box-shadow: 0 0 0 0 rgba(167,139,250,0); }
    }
    .processing-card {
        animation: pulse-glow 1.8s infinite;
    }

    /* ── Streamlit widget overrides ── */
    .stFileUploader label { color: #a78bfa !important; font-weight: 600; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(124,58,237,0.5) !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(124,58,237,0.5) !important;
        border-radius: 16px !important;
        background: rgba(124,58,237,0.04) !important;
    }
    .stAudio { border-radius: 12px; overflow: hidden; }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #334155;
        font-size: 0.78rem;
        margin-top: 3rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: lazy-load heavy dependencies so Streamlit starts fast
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load and cache the Whisper model (runs once per session)."""
    import whisper
    return whisper.load_model("base")   # "base" is fast; swap to "small" for accuracy


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Write audio bytes to a temp file, run Whisper, return transcript.

    Args:
        audio_bytes: Raw audio file content.
        suffix:      File extension (.wav, .mp3, etc.)

    Returns:
        Transcribed text string.
    """
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path, fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tone → display metadata
# ─────────────────────────────────────────────────────────────────────────────
TONE_META = {
    "angry":   {"icon": "😡", "css": "tone-angry",   "label": "Angry"},
    "happy":   {"icon": "😊", "css": "tone-happy",   "label": "Happy"},
    "neutral": {"icon": "😐", "css": "tone-neutral", "label": "Neutral"},
    "sad":     {"icon": "😢", "css": "tone-sad",     "label": "Sad"},
    "excited": {"icon": "🤩", "css": "tone-excited", "label": "Excited"},
}


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎙️ VibeSync</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Record a voice message · We detect your vibe · AI rephrases it perfectly</p>',
    unsafe_allow_html=True,
)

# ── API Key check ─────────────────────────────────────────────────────────────
google_key = os.getenv("GOOGLE_API_KEY", "").strip()
groq_key   = os.getenv("GROQ_API_KEY",   "").strip()
_no_google = not google_key or google_key == "your_google_api_key_here"
_no_groq   = not groq_key   or groq_key   == "your_groq_api_key_here"

if _no_google and _no_groq:
    st.error(
        "🔑  **No LLM API key found.** Add at least one of these to your `.env` file:\n"
        "- `GOOGLE_API_KEY` — [Get Gemini key](https://aistudio.google.com)\n"
        "- `GROQ_API_KEY`   — [Get Groq key](https://console.groq.com)",
        icon="🚨",
    )
elif _no_google:
    st.info(
        "ℹ️  Gemini key not found — **Groq (llama-3.1-8b-instant)** will be used as the LLM.",
        icon="🤖",
    )

# ── Step 1: Audio Input (Record or Upload) ────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 1 — Give Us Your Voice</p>', unsafe_allow_html=True)

from audio_recorder_streamlit import audio_recorder

tab_record, tab_upload = st.tabs(["🎙️  Record Now", "📁  Upload File"])

audio_bytes: bytes | None = None
audio_suffix: str = ".wav"

with tab_record:
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0.8rem;'>"
        "Click the mic button to start recording. Click again to stop. "
        "Whisper runs locally — your audio never leaves your device."
        "</p>",
        unsafe_allow_html=True,
    )
    # audio_recorder returns raw WAV bytes after recording, or None
    recorded_bytes = audio_recorder(
        text="",
        recording_color="#ef4444",
        neutral_color="#a78bfa",
        icon_name="microphone",
        icon_size="3x",
        pause_threshold=3.0,   # auto-stop after 3 s of silence
        sample_rate=16_000,    # Whisper prefers 16 kHz
    )
    if recorded_bytes:
        audio_bytes  = recorded_bytes
        audio_suffix = ".wav"
        st.audio(recorded_bytes, format="audio/wav")
        st.success("✅  Recording captured! Hit \"Analyze & Rephrase\" below.")

with tab_upload:
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0.8rem;'>"
        "Upload a .wav, .mp3, .m4a, .ogg or .flac voice note."
        "</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        audio_bytes  = uploaded_file.read()
        audio_suffix = "." + uploaded_file.name.split(".")[-1]
        st.audio(audio_bytes, format=f"audio/{audio_suffix.lstrip('.')}")

st.markdown("</div>", unsafe_allow_html=True)

# ── Step 2: Process ────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 2 — Sync Your Vibe</p>', unsafe_allow_html=True)

process_btn = st.button("🚀  Analyze & Rephrase", disabled=(audio_bytes is None))

st.markdown("</div>", unsafe_allow_html=True)

# ── Processing logic ───────────────────────────────────────────────────────────
if process_btn and audio_bytes:

    from pipeline import mock_tone_analysis, rephrase_with_vibe

    # ─── Transcription ────────────────────────────────────────────────────────
    with st.spinner("🎙️  Transcribing audio with Whisper…"):
        transcript = transcribe_audio(audio_bytes, suffix=audio_suffix)


    if not transcript:
        st.error("❌ Whisper couldn't extract any speech. Try a clearer audio file.")
        st.stop()

    # ─── Tone detection (mock) ────────────────────────────────────────────────
    with st.spinner("🧠  Analyzing emotional tone…"):
        time.sleep(0.6)  # small artificial delay so the spinner is visible
        detected_tone = mock_tone_analysis(audio_path=None)

    # ─── LangChain LCEL pipeline (Gemini primary → Groq fallback) ──────────────
    with st.spinner("✨  VibeSync is rephrasing your message…"):
        rephrased, llm_name = rephrase_with_vibe(transcript, detected_tone)

    tone_info = TONE_META.get(detected_tone, TONE_META["neutral"])

    # ─── Results ──────────────────────────────────────────────────────────────
    st.success("✅  Vibe synced successfully!")

    # Transcript card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📝 Whisper Transcript</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tone card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🎭 Detected Emotional Tone</p>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="tone-badge {tone_info["css"]}">'
        f'{tone_info["icon"]}  {tone_info["label"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<br><small style='color:#64748b; font-size:0.78rem;'>Powered by mock tone analyzer "
        "(MVP stub — replace with Hume AI in production)</small>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Rephrased output card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🪄 VibeSync Output — Tone-Enhanced Message</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="output-box">{rephrased}</div>', unsafe_allow_html=True)
    st.markdown(
        f"<br><small style='color:#64748b; font-size:0.78rem;'>"
        f"Generated by <strong style='color:#a78bfa'>{llm_name}</strong> "
        f"via LangChain LCEL pipeline</small>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Copy helper
    st.code(rephrased, language=None)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="footer">VibeSync MVP · Powered by Whisper · LangChain · Gemini 2.5 Flash + Groq Fallback</p>',
    unsafe_allow_html=True,
)
