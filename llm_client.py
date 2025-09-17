# llm_client.py
"""
Wrapper for interacting with a local Ollama model (e.g., TinyLlama, Phi).
Focuses on providing concise financial answers in natural language,
correctly formatted numbers, and readable responses for Streamlit.
"""

from typing import List, Dict, Any
import ollama # type: ignore
import logging
import re
from utils import parse_amount_to_number  # optional: for future numeric handling

# Change to whichever model you have pulled, e.g., "tinyllama" or "phi"
OLLAMA_MODEL = "gemma:2b-instruct"


def chat_with_ollama(messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
    """
    Send a conversation to a local Ollama model and return a clean response.

    Parameters
    ----------
    messages : list of dict
        Each dict = {"role": "system"|"user"|"assistant", "content": str}
    stream : bool
        If True, return a generator of chunks; else return final dict.

    Returns
    -------
    dict
        {
            "answer": "clean text answer",
            "numbers": ["12.3", "456000"],
            "raw": "raw text from model"
        }
    """
    try:
        # Strengthen system prompt if present
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += (
                "\n\nGuidelines for answering:\n"
                "- Analyze the financial data carefully; do not copy raw text.\n"
                "- Answer concisely in plain natural language.\n"
                "- Present all key financial figures as plain numbers or $ values.\n"
                "- Correctly interpret symbols, commas, percentages, and units.\n"
                "- Avoid markdown formatting, unnecessary headings or lists.\n"
                "- Make answers easy to understand for non-experts.\n"
                "- Use proper numeric formatting (e.g., $1,250,000; 40%).\n"
            )

        # Query the local Ollama model
        raw_resp = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=stream)

        # If streaming, return generator directly
        if stream:
            return raw_resp

        # Extract content safely
        content_text = ""
        if isinstance(raw_resp, dict):
            content_text = raw_resp.get("message", {}).get("content", "") or ""
        else:
            content_text = str(raw_resp)

        # Sanitize for Streamlit
        clean_text = _sanitize_for_streamlit(content_text)

        # Extract numbers and currency patterns
        numbers = re.findall(r"(?:\$)?\d+(?:,\d{3})*(?:\.\d+)?%?", clean_text)

        return {
            "answer": clean_text,
            "numbers": numbers,
            "raw": content_text
        }

    except Exception as e:
        logging.exception("Ollama error")
        return {
            "answer": f"[Error] Could not contact local model: {e}",
            "numbers": [],
            "raw": ""
        }


def _sanitize_for_streamlit(text: str) -> str:
    """
    Remove or neutralize characters that Streamlit's markdown might misinterpret.
    """
    # Remove headings
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    # Remove bold/underline markers
    text = text.replace("**", "").replace("__", "")
    # Replace bullet lists with simple bullets
    text = text.replace("* ", "• ")
    return text.strip()
