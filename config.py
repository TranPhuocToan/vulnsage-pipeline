"""
config.py – Cấu hình trung tâm cho toàn bộ hệ thống VulnSage Pipeline.

Tất cả các tham số inference được cố định (Invariant I3) để đảm bảo
tính công bằng trong Ablation Study giữa P0, P1 và P2.
"""

import os
try:
    from dotenv import load_dotenv
except ImportError:  # Offline unit-test environments need no .env loader.
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# ===== MODEL CONFIGURATION =====
# Candidates: gemini-2.5-flash, gpt-4o, qwen3-coder
MODEL_NAME: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
API_KEY: str = os.getenv("API_KEY", "")
API_BASE_URL: str = os.getenv("API_BASE_URL", "")

# ===== INFERENCE PARAMETERS (FIXED - KHÔNG THAY ĐỔI GIỮA P0/P1/P2) =====
# Invariant I1: temperature = 0.0 trên MỌI lệnh gọi LLM
TEMPERATURE: float = 0.0
MAX_OUTPUT_TOKENS: int = 2048
MAX_RETRY: int = 3

# ===== DATA CONFIGURATION =====
DATA_RAW_PATH: str = "data/raw/vulnsage_g3.json"
DATA_PATH: str = "data/processed/parsed_g3.json"
PILOT_PATH: str = "data/pilot/pilot_20.json"
PILOT_SIZE: int = 20
MAIN_EXPERIMENT_SIZE: int = 150

# ===== RESULTS CONFIGURATION =====
RESULTS_DIR: str = "results/"
RESULTS_PILOT_DIR: str = "results/pilot/"
RESULTS_MAIN_DIR: str = "results/main/"
RESULTS_INTERMEDIATE_DIR: str = "results/intermediate/"
