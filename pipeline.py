"""
pipeline.py — VibeSync LangChain LCEL Backend
==============================================

This module contains:
  1. mock_tone_analysis()   — stub that randomly returns a tone label
  2. build_vibesync_chain() — LCEL chain: PromptTemplate | LLM | StrOutputParser
  3. rephrase_with_vibe()   — single entry-point used by app.py

HOW THE LCEL PIPELINE WORKS
────────────────────────────
LangChain Expression Language (LCEL) lets you compose pipeline steps with the
pipe operator (|).  Each step is a Runnable, and output of one step becomes the
input of the next.

    PromptTemplate  →  ChatGoogleGenerativeAI  →  StrOutputParser
         (1)                    (2)                      (3)

Step 1 – PromptTemplate:
  Accepts a dict  {"transcribed_text": ..., "detected_tone": ...}
  and renders a fully-formed prompt string that instructs the LLM.

Step 2 – ChatGoogleGenerativeAI (gemini-2.5-flash):
  Receives the rendered prompt, calls the Gemini API, returns an AIMessage.

Step 3 – StrOutputParser:
  Unwraps the AIMessage and returns a plain Python string — the final output.
"""

import os
import random
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # reads GOOGLE_API_KEY from .env


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
# 2.  LCEL CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# The prompt template uses two input variables:
#   {transcribed_text} — raw transcript from Whisper
#   {detected_tone}    — tone label from mock_tone_analysis()
#   {emojis}           — resolved from TONE_EMOJI_MAP inside rephrase_with_vibe()
VIBESYNC_PROMPT_TEMPLATE = """
You are VibeSync, an expressive communication assistant.

The user spoke a voice message that was transcribed. You must rephrase the
message so it perfectly matches the detected emotional tone, and naturally
weave in relevant emojis throughout the text.

Transcribed message:
"{transcribed_text}"

Detected emotional tone: {detected_tone}
Tone emojis to use: {emojis}

Rephrasing rules based on tone:
- angry   : Use ALL CAPS for emphasis, short punchy sentences, include 😡🔥💢
- happy   : Warm, enthusiastic language, exclamation points, include 😊🎉✨
- neutral : Clear, professional tone, measured language, include 😐💬🙂
- sad     : Soft, empathetic words, ellipses for pauses, include 😢💧🥺
- excited : Energetic, fast-paced, lots of emphasis, include 🤩🚀⚡

Output ONLY the rephrased message — no explanations, no preamble.
""".strip()


def build_vibesync_chain():
    """
    Constructs and returns the LCEL chain.

    Chain anatomy:
      prompt_template  →  llm  →  output_parser
          Runnable         Runnable      Runnable
    The pipe operator (|) wires them together.  When .invoke() is called,
    execution flows left-to-right automatically.

    Returns:
        A compiled LCEL Runnable that accepts:
            {"transcribed_text": str, "detected_tone": str, "emojis": str}
        and returns:
            str  (the rephrased message)
    """
    # Step 1 – PromptTemplate
    # input_variables tells LangChain which keys to expect in the dict
    # passed to .invoke()
    prompt = PromptTemplate(
        input_variables=["transcribed_text", "detected_tone", "emojis"],
        template=VIBESYNC_PROMPT_TEMPLATE,
    )

    # Step 2 – LLM  (gemini-2.5-flash via langchain-google-genai)
    # temperature=0.8 adds creativity while keeping responses coherent
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Step 3 – Output parser
    # StrOutputParser.invoke() calls output.content on the AIMessage
    # returned by the LLM, giving us a plain Python string.
    output_parser = StrOutputParser()

    # Wire the chain with LCEL's pipe operator
    #   dict input → prompt renders it → LLM generates → parser extracts text
    chain = prompt | llm | output_parser

    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def rephrase_with_vibe(transcribed_text: str, detected_tone: str) -> str:
    """
    Main entry-point called by app.py.

    Resolves the emoji palette for the detected tone, builds the LCEL chain,
    invokes it, and returns the final rephrased string.

    Args:
        transcribed_text: Raw transcript from Whisper.
        detected_tone:    Tone label (e.g. "angry", "happy", ...).

    Returns:
        Rephrased message string from Gemini.
    """
    emojis = TONE_EMOJI_MAP.get(detected_tone, "💬")

    # Build a fresh chain for each call (stateless; no memory needed in MVP)
    chain = build_vibesync_chain()

    # .invoke() kicks off the pipeline:
    #   1. PromptTemplate fills {transcribed_text}, {detected_tone}, {emojis}
    #   2. ChatGoogleGenerativeAI sends the rendered prompt to Gemini API
    #   3. StrOutputParser returns the plain text response
    result: str = chain.invoke(
        {
            "transcribed_text": transcribed_text,
            "detected_tone":    detected_tone,
            "emojis":           emojis,
        }
    )

    return result
