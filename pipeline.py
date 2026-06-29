"""
pipeline.py — VibeSync LangChain LCEL Backend
==============================================

This module contains:
  1. mock_tone_analysis()   — stub that randomly returns a tone label
  2. get_llm()              — smart LLM selector: tries Gemini first, falls back to Groq
  3. build_vibesync_chain() — LCEL chain: PromptTemplate | LLM | StrOutputParser
  4. rephrase_with_vibe()   — single entry-point used by app.py

HOW THE LCEL PIPELINE WORKS
────────────────────────────
LangChain Expression Language (LCEL) lets you compose pipeline steps with the
pipe operator (|).  Each step is a Runnable, and output of one step becomes the
input of the next.

    PromptTemplate  →  LLM (Gemini or Groq)  →  StrOutputParser
         (1)                   (2)                      (3)

Step 1 – PromptTemplate:
  Accepts a dict  {"transcribed_text": ..., "detected_tone": ..., "emojis": ...}
  and renders a fully-formed prompt string that instructs the LLM.

Step 2 – LLM (with automatic fallback):
  PRIMARY  → ChatGoogleGenerativeAI (gemini-2.5-flash)  if GOOGLE_API_KEY is set
  FALLBACK → ChatGroq (llama-3.1-8b-instant)            if GROQ_API_KEY is set
  The fallback kicks in automatically when Gemini is unavailable or its key is missing.

Step 3 – StrOutputParser:
  Unwraps the AIMessage and returns a plain Python string — the final output.
"""

import os
import random
import logging
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads GOOGLE_API_KEY and GROQ_API_KEY from .env

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MOCK TONE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Tone → emoji palette mapping used by the prompt
TONE_EMOJI_MAP = {
    "angry":   "😡🔥💢",
    "happy":   "😊🎉✨",
    "neutral": "😐💬🙂",
    "sad":     "😢💧🥺",
    "excited": "🤩🚀⚡",
}


def mock_tone_analysis(audio_path: str | None = None) -> str:
    """
    Stub tone detector — returns a random tone label.

    In a production build this function would call the Hume AI Speech
    Prosody API (or a similar service) and return the dominant emotion.
    For the MVP we randomly sample from a weighted distribution so that
    testing covers all branches of the prompt logic.

    Args:
        audio_path: Path to the recorded .wav / .mp3 file (ignored in stub).

    Returns:
        One of: "angry", "happy", "neutral", "sad", "excited"
    """
    tones = list(TONE_EMOJI_MAP.keys())
    # Weighted so "angry" and "happy" appear more often during demos
    weights = [0.25, 0.35, 0.20, 0.10, 0.10]
    detected = random.choices(tones, weights=weights, k=1)[0]
    return detected


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SMART LLM SELECTOR  (Gemini primary → Groq fallback)
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.8):
    """
    Returns the best available LLM using this priority order:

      1. Gemini 2.5 Flash  — if GOOGLE_API_KEY is present and valid
      2. Groq llama-3.1-8b — if GROQ_API_KEY is present (automatic fallback)
      3. Raises RuntimeError if neither key is available

    Both LLMs implement the same LangChain BaseChatModel interface, so
    the LCEL chain works identically regardless of which one is selected.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        Tuple[BaseChatModel, str]: (llm_instance, human_readable_name)

    Raises:
        RuntimeError: When no API key is configured.
    """
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    groq_key   = os.getenv("GROQ_API_KEY",   "").strip()

    # ── PRIMARY: Google Gemini ────────────────────────────────────────────────
    if google_key and google_key != "your_google_api_key_here":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=temperature,
                google_api_key=google_key,
            )
            logger.info("LLM selected: Gemini 2.5 Flash (primary)")
            return llm, "Gemini 2.5 Flash"
        except Exception as e:
            logger.warning("Gemini init failed (%s) — trying Groq fallback.", e)

    # ── FALLBACK: Groq ────────────────────────────────────────────────────────
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.1-8b-instant",   # fast & free-tier friendly
                temperature=temperature,
                groq_api_key=groq_key,
            )
            logger.info("LLM selected: Groq llama-3.1-8b-instant (fallback)")
            return llm, "Groq llama-3.1-8b-instant"
        except Exception as e:
            logger.warning("Groq init failed (%s).", e)

    raise RuntimeError(
        "No LLM available. Set GOOGLE_API_KEY (Gemini) or GROQ_API_KEY (Groq) in your .env file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LCEL CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# The prompt template receives three variables injected by rephrase_with_vibe():
#   {transcribed_text} — raw transcript from Whisper
#   {detected_tone}    — tone label from mock_tone_analysis()
#   {emojis}           — resolved from TONE_EMOJI_MAP
VIBESYNC_PROMPT_TEMPLATE = """
You are VibeSync, a subtle tone-enhancement assistant.

The user recorded a voice message. Your job is to make MINIMAL edits to the
transcribed text so it clearly communicates the same message while also
reflecting the emotional tone. You must NOT rewrite or restructure the sentence.

Transcribed message (keep this mostly intact):
"{transcribed_text}"

Detected emotional tone: {detected_tone}
Tone emojis to sprinkle in: {emojis}

Your rules:
1. Preserve the original words as much as possible (aim for 80-90% same).
2. Only adjust: emphasis (CAPS for key words), punctuation, a word swap here
   or there, and 1-3 tone-appropriate emojis placed naturally in the text.
3. Do NOT change the meaning, add new ideas, or rewrite full sentences.
4. Tone-specific touches (keep them subtle):
   - angry   : Capitalise the most important word(s). Maybe end with "!" Add 😡 or 🔥 once.
   - happy   : Add a "!" or "😊". Maybe swap one word for a warmer synonym.
   - neutral : Light clean-up only. One 💬 or 🙂 at most.
   - sad     : Soften one word slightly. Add "..." for a pause. Add 😢 once.
   - excited : Add a "!" or "!!", emphasise one word. Add 🤩 or ⚡ once.
5. Output ONLY the lightly enhanced message — no labels, no explanation.
""".strip()


def build_vibesync_chain(temperature: float = 0.8):
    """
    Constructs and returns the LCEL chain plus the active LLM name.

    Chain anatomy:
      prompt_template  →  llm  →  output_parser
          Runnable         Runnable      Runnable
    The pipe operator (|) wires them together. When .invoke() is called,
    execution flows left-to-right automatically.

    Returns:
        Tuple[chain, llm_name]:
          chain    — LCEL Runnable that accepts the input dict and returns str
          llm_name — human-readable name of the selected LLM (for UI display)
    """
    # Step 1 – PromptTemplate
    # input_variables tells LangChain which keys to expect in .invoke()
    prompt = PromptTemplate(
        input_variables=["transcribed_text", "detected_tone", "emojis"],
        template=VIBESYNC_PROMPT_TEMPLATE,
    )

    # Step 2 – LLM  (Gemini primary, Groq fallback)
    llm, llm_name = get_llm(temperature=temperature)

    # Step 3 – Output parser
    # StrOutputParser extracts the plain string from the AIMessage returned
    # by any BaseChatModel — works identically for Gemini and Groq.
    output_parser = StrOutputParser()

    # Wire the chain with LCEL's pipe operator:
    #   dict → PromptTemplate renders it → LLM generates → parser extracts text
    chain = prompt | llm | output_parser

    return chain, llm_name


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def rephrase_with_vibe(transcribed_text: str, detected_tone: str) -> tuple[str, str]:
    """
    Main entry-point called by app.py.

    Resolves the emoji palette for the detected tone, builds the LCEL chain
    (with automatic Gemini → Groq fallback), invokes it, and returns the
    rephrased string together with the name of the LLM that was used.

    Args:
        transcribed_text: Raw transcript from Whisper.
        detected_tone:    Tone label (e.g. "angry", "happy", ...).

    Returns:
        Tuple[str, str]: (rephrased_message, llm_name_used)
    """
    emojis = TONE_EMOJI_MAP.get(detected_tone, "💬")

    # Build a fresh chain for each call (stateless; no memory needed in MVP).
    # llm_name is returned so the UI can display which LLM was used.
    chain, llm_name = build_vibesync_chain()

    # .invoke() kicks off the pipeline:
    #   1. PromptTemplate fills {transcribed_text}, {detected_tone}, {emojis}
    #   2. LLM (Gemini or Groq) sends the rendered prompt and gets a response
    #   3. StrOutputParser returns the plain text
    result: str = chain.invoke(
        {
            "transcribed_text": transcribed_text,
            "detected_tone":    detected_tone,
            "emojis":           emojis,
        }
    )

    return result, llm_name
