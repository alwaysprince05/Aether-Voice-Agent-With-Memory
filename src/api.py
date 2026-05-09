"""FastAPI server for the Voice AI Agent Web App."""

import os
import uuid
from typing import Dict, Any
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import tempfile

from src.config import config
from src.todo_manager import ToDoManager
from src.memory_system import MemorySystem
from src.agent_core import AgentCore

from gtts import gTTS
import io
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="AETHER Voice AI Agent API")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health_check():
    return {"status": "online", "agent": "AETHER"}

@app.get("/api/tts")
async def get_tts(text: str):
    """Generate TTS audio from text and stream it."""
    try:
        tts = gTTS(text=text, lang='en', tld='com') # 'com' for US English
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent Components globally
openai_client = OpenAI(
    api_key=config.groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

todo_manager = ToDoManager(
    storage_path=f"{config.data_dir}/todos.json"
)
memory_system = MemorySystem(
    storage_path=f"{config.data_dir}/memories.json"
)
agent_core = AgentCore(
    openai_client=openai_client,
    todo_manager=todo_manager,
    memory_system=memory_system
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "web_default"

import base64

def text_to_base64_audio(text: str) -> str:
    """Generate TTS MP3 and return as base64 string."""
    try:
        if not text or len(text.strip()) == 0:
            return ""
        # Limit text length to prevent huge responses
        short_text = text[:800] 
        tts = gTTS(text=short_text, lang='en', tld='com')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response = agent_core.process_input(req.message, session_id=req.session_id)
        audio_b64 = text_to_base64_audio(response.text)
        
        return {
            "text": response.text,
            "audio_b64": audio_b64,
            "success": response.success,
            "error": response.error,
            "tool_calls": [tc.tool_name for tc in response.tool_calls]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/voice")
async def voice_endpoint(audio: UploadFile = File(...), session_id: str = "web_default"):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        try:
            # Transcribe
            with open(tmp_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    response_format="text"
                )
            
            # Process text
            response = agent_core.process_input(transcription, session_id=session_id)
            audio_b64 = text_to_base64_audio(response.text)
            
            return {
                "transcription": transcription,
                "text": response.text,
                "audio_b64": audio_b64,
                "success": response.success,
                "error": response.error,
                "tool_calls": [tc.tool_name for tc in response.tool_calls]
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/todos")
async def get_todos():
    items = todo_manager.list_todos()
    return [{"id": i.id, "description": i.description, "status": i.status} for i in items]

@app.get("/api/memories")
async def get_memories():
    # Helper to get all memories. We can just list the directory or access internal state.
    # Since MemorySystem doesn't have list_all natively, we can search with empty query
    # or expose a method. For now we will return dummy if we can't access easily.
    try:
        # Accessing private dict for the sake of UI dashboard
        memories = list(memory_system._memories.values())
        return [{"id": m.id, "content": m.content, "tags": m.tags, "timestamp": m.timestamp.isoformat()} for m in memories]
    except Exception as e:
        return []

# Mount frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
