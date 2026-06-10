"""
Central configuration for the FairSearch-arXiv baseline pipeline.
Edit paths/params here; every script imports from this file.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = str(ROOT / "chroma_store")

# Raw Kaggle file (download instructions in README). One JSON object per line.
RAW_SNAPSHOT = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

# Outputs of each stage
SAMPLE_FILE = DATA_DIR / "sample_50k.jsonl"      # stratified sample
QUERIES_FILE = DATA_DIR / "queries.jsonl"        # query set + relevance judgments (qrels)

# ---- Sampling --------------------------------------------------------
SAMPLE_SIZE = 50_000
CATEGORY_PREFIX = "cs."   # keep only primary category starting with this; set "" to keep all
YEAR_MIN, YEAR_MAX = 2020, 2025
RANDOM_SEED = 42

# ---- Evaluation ------------------------------------------------------
N_QUERIES = 200           # number of held-out query papers
K_VALUES = [5, 10]        # P@k / R@k cutoffs to report
QUERY_METHOD = "heuristic"  # "heuristic" (zero-dependency) or "llm" (needs API key)

# ---- Models ----------------------------------------------------------
MINILM_MODEL = "all-MiniLM-L6-v2"
SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"
EMBED_BATCH = 32

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
