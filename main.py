from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from translation import load_model_and_tokenizer, generate_translation
from pydantic import BaseModel
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
)

# Load model and tokenizer at startup
load_model_and_tokenizer()

# REST API Input Schema
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    """
    REST API endpoint for translation.
    """
    try:
        logger.info(f"Translation request received: {request}")
        translation = generate_translation(
            request.text, request.source_language, request.target_language
        )
        return {"translated_text": translation}
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time translation.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            # Receive data from WebSocket client
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")

            text = data.get("text")
            src_lang = data.get("source_language")
            tgt_lang = data.get("target_language")

            if not text or not src_lang or not tgt_lang:
                await websocket.send_json(
                    {"error": "Invalid input. Please provide text, source_language, and target_language."}
                )
                continue

            # Generate translation
            translation = generate_translation(text, src_lang, tgt_lang)
            await websocket.send_json({"translated_text": translation})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
