import os
import json
import requests
import pyarrow.parquet as pq
import io
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

# ── Config ──────────────────────────────────────────────
API_KEY = "psychoxd"  # 🔥 Hardcoded — environment ki zaroorat nahi
DEVELOPER = "@psychopathmc"
BASE_URL = "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main"

app = FastAPI(title="ICMR + HITEK Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: fetch & filter Parquet ────────────────────
def fetch_filter(url: str, column: str, value: str, limit: int = 5):
    try:
        resp = requests.get(url, timeout=10)  # ⏱️ 10 sec timeout
        if resp.status_code != 200:
            return []
        table = pq.read_table(io.BytesIO(resp.content))
        df = table.to_pandas()
        if column not in df.columns:
            return []
        filtered = df[df[column] == value]
        return filtered.head(limit).to_dict(orient="records")
    except Exception as e:
        print(f"Error in fetch_filter: {e}")
        return []

# ── Endpoints ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {"phone": True, "aadhar": True},
        "index_source": "remote",
        "columns": ["name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber", "address", "district", "pincode", "state", "town", "source"],
        "docs": "/docs",
        "developer": DEVELOPER,
    }

@app.get("/health")
def health():
    return {"status": "ok", "developer": DEVELOPER}

@app.get("/search")
def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    key: str = Query(None),  # Optional now, but we check
    limit: int = Query(5, ge=1, le=20),  # 🔥 Limit 5 for speed
):
    # 🔥 Key check — hardcoded
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    query = (q or mobile or "").strip()
    if not query:
        raise HTTPException(422, "Provide q or mobile")

    # Shard based on last digit
    last_digit = query[-1]
    shard = int(last_digit) % 7

    # Try phone index first
    phone_url = f"{BASE_URL}/idx_phone.{shard}.parquet"
    results = fetch_filter(phone_url, "phoneNumber", query, limit)

    # If no phone, try aadhar
    if not results:
        aadhar_url = f"{BASE_URL}/idx_aadhar.{shard}.parquet"
        results = fetch_filter(aadhar_url, "aadharNumber", query, limit)

    return {
        "success": len(results) > 0,
        "query": query,
        "count": len(results),
        "results": results,
        "developer": DEVELOPER,
    }
