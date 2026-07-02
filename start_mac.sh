#!/bin/bash
export GEO_CHAT_MODEL=qwen3.5:4b
export GEO_VISION_MODEL=moondream
python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
