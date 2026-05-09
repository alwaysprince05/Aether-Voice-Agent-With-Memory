# 🌌 AETHER · Voice AI Agent with Memory

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-f3d122?style=for-the-badge&logo=ai)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**AETHER** is a premium, full-stack Voice AI Agent designed for high-performance productivity. It combines a stunning **Glassmorphic Bento Grid Dashboard** with powerful **Long-Term Memory** and **Task Management**, all powered by the lightning-fast Groq LPU inference engine.

---

## ✨ Key Features

### 📐 Premium Bento Grid Interface
- **Modern Dashboard**: A modular, responsive Bento Grid layout inspired by 2025 design trends.
- **Glassmorphism & Neumorphism**: Subtle frosted glass effects, depth-driven shadows, and premium "claymorphic" elements.
- **3D Animations**: Hardware-accelerated CSS 3D objects (spinning cubes, orbiting rings) and ambient gradient orbs for a "living" UI.

### 🧠 Intelligent Core
- **Long-Term Memory**: Automatically stores user facts, preferences, and schedules. It "recalls" this information contextually during conversations.
- **Natural Task Management**: Create, list, update, and complete tasks using natural language or voice commands.
- **Voice-First Design**: Seamless browser-based microphone integration with real-time **EQ Equalizer** and **Ripple Animations**.

### ⚡ Performance & Scale
- **Groq Powered**: Uses `Llama-3-70b` for reasoning and `Whisper-large-v3` for sub-second transcription.
- **FastAPI Backend**: Robust asynchronous API handling both chat and multipart audio streams.
- **Docker Ready**: Fully containerized for one-command deployment.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **AI Models** | Llama 3 (Inference), Whisper-large-v3 (STT) |
| **Backend** | Python 3.14+, FastAPI, Uvicorn |
| **Frontend** | Vanilla JavaScript (ES6+), CSS3 (Grid & 3D), HTML5 |
| **Infrastructure** | Docker, Docker Compose |
| **Tools** | Groq LPU, OpenAI SDK |

---

## 🚀 Quick Start

### 1. Prerequisites
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/))

### 2. Deployment with Docker (Recommended)
The easiest way to run AETHER is using Docker:

```bash
# Clone the repository
git clone https://github.com/Aerospace-prog/Aether-Voice-Agent-With-Memory.git
cd Aether-Voice-Agent-With-Memory

# Configure your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Launch the system
docker-compose up -d
```
Access the dashboard at **[http://localhost:8000](http://localhost:8000)**.

### 3. Local Development Setup
If you prefer to run locally without Docker:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY=your_key_here

# Start the server
uvicorn src.api:app --reload --port 8000
```

---

## 📖 Usage Guide

### Voice Mode
1. Click the **Microphone** icon in the dashboard.
2. Grant microphone permissions in your browser.
3. Speak a command (e.g., *"Remember that I have a meeting tomorrow at 3pm"*).
4. Click the mic again to stop. AETHER will transcribe, process, and update the UI live.

### Quick Actions
Use the **Quick Actions** bento cell for one-click access to:
- **Recall**: Ask AETHER to summarize what it knows about you.
- **View Tasks**: Instantly list all pending to-dos.
- **Summarize**: Get a snapshot of your current productivity state.

---

## 📁 Project Structure

```text
Aether-Voice-Agent-With-Memory/
├── src/
│   ├── api.py           # FastAPI Web Server
│   ├── agent_core.py    # LLM & Tool-Calling Logic
│   ├── memory_system.py # Memory Storage & Recall
│   ├── todo_manager.py  # Task Management CRUD
│   └── config.py        # Env & Config Management
├── frontend/
│   ├── index.html       # Bento Grid UI
│   ├── style.css        # Glassmorphic & 3D Styling
│   └── app.js           # Frontend Audio & API Logic
├── Dockerfile           # Production Container Build
└── docker-compose.yml   # Multi-service Orchestration
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
**AETHER** — *The future of personal AI, one memory at a time.*
