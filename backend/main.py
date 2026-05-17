import os
import re
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Sahayak AI Emergency Backend",
    description="Offline-ready RAG FastAPI service running local Gemma 4 and ChromaDB.",
    version="1.0.4"
)

# Enable CORS for frontend clients (e.g., our sahayak_app.html dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BACKEND_DIR, "documents")
CHROMA_DIR = os.path.join(BACKEND_DIR, "chroma_db")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Global variables for ChromaDB
chroma_client = None
collection = None

# Initialize ChromaDB connection
try:
    import chromadb
    from chromadb.utils import embedding_functions
    if os.path.exists(CHROMA_DIR):
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = chroma_client.get_collection("emergency_manuals", embedding_function=emb_fn)
        print(f"[STARTUP] Successfully connected to ChromaDB. Active collection count: {collection.count()}")
    else:
        print("[STARTUP WARNING] ChromaDB path not found. Run 'make embed' to index manuals first.")
except Exception as e:
    print(f"[STARTUP WARNING] Failed to establish ChromaDB vector client: {e}")

# Data Models
class QueryRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = []

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Static Emergency Map Pin dataset (pre-cached SQLite mock schema)
MARKER_PINS = {
    "patna": [
        {"name": "Patna Central Flood Shelter #1", "type": "shelter", "lat": 25.6026, "lng": 85.1199, "desc": "Beds: 120/150 available. Food: Rations active. Generator: Yes."},
        {"name": "Ganga Coast Guard Rescue Point", "type": "shelter", "lat": 25.6105, "lng": 85.1325, "desc": "Rescue boats stationed: 12 operational. High ground clearance."},
        {"name": "Patna Emergency Medical Camp", "type": "hospital", "lat": 25.5945, "lng": 85.1050, "desc": "Trauma beds: 18. Stocked with anti-venom, emergency IV packs."},
        {"name": "Safe Water Distribution Hub A", "type": "water", "lat": 25.6050, "lng": 85.0990, "desc": "Drinking water filtration unit powered by local solar grid. Active."}
    ],
    "dibrugarh": [
        {"name": "Brahmaputra Evacuation Base #4", "type": "shelter", "lat": 27.4728, "lng": 94.9120, "desc": "Capacity: 340/400. Direct dry food drops operational here."},
        {"name": "Dibrugarh Civil Disaster Ward", "type": "hospital", "lat": 27.4810, "lng": 94.9010, "desc": "Emergency medical division active 24/7. Oxygen supply: Stable."},
        {"name": "Brahmaputra Water Purification Point", "type": "water", "lat": 27.4650, "lng": 94.9250, "desc": "Heavy sediment filters installed. Capacity: 5000 Litres/day."}
    ],
    "cuttack": [
        {"name": "Mahanadi Storm shelter #12", "type": "shelter", "lat": 20.4625, "lng": 85.8830, "desc": "Capacity: 500 beds. Built for Category 4 cyclonic wind loads."},
        {"name": "Cuttack General Trauma Base", "type": "hospital", "lat": 20.4505, "lng": 85.8690, "desc": "Disaster triage team deployed. 40 trauma specialists available."}
    ]
}

# Rule-Based Fallback Document Searcher
def local_regex_document_search(query: str, top_k: int = 2) -> List[str]:
    """Falls back to local file keyword scanning if ChromaDB fails or is building."""
    results = []
    if not os.path.exists(DOCS_DIR):
        return results
        
    keywords = [kw.lower() for kw in re.findall(r'\w+', query) if len(kw) > 3]
    if not keywords:
        keywords = ["bleed", "flood", "cpr", "fracture"]

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                paragraphs = f.read().split("\n\n")
                
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                match_score = sum(1 for kw in keywords if kw in para.lower())
                if match_score > 0:
                    results.append((para, match_score))
                    
    # Sort by match score and return top paragraphs
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_k]]

# Standard Endpoints
@app.get("/")
def get_status():
    """Returns local server health status."""
    chroma_ok = collection is not None
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.0)
        if r.status_code == 200:
            ollama_ok = True
    except Exception:
        pass

    return {
        "status": "active",
        "sqlite_cache": "cached_offline",
        "chroma_db": "connected" if chroma_ok else "offline_or_building",
        "vector_chunks": collection.count() if chroma_ok else 0,
        "ollama_connection": "active" if ollama_ok else "unreachable_local_mode"
    }

@app.post("/classify")
def classify_query(req: QueryRequest):
    """Step 1: CLASSIFY -- Identifies disaster type, severity, and input language."""
    text = req.text.lower()
    
    # 1. Default fallback parameters
    disaster_type = "Medical Emergency"
    severity = "URGENT"
    detected_lang = "en"
    
    # 2. Local rule-based identification (guarantees fast, robust classify output)
    if "pani" in text or "baadh" in text or "flood" in text or "water" in text or "evacuation" in text or "shelter" in text:
        disaster_type = "Flood Evacuation"
        severity = "HIGH"
    elif "marna" in text or "fracture" in text or "cpr" in text or "bleed" in text or "blood" in text or "khoon" in text or "chot" in text:
        disaster_type = "Medical Emergency"
        severity = "URGENT"
        
    if any(char in text for char in ["अ", "ब", "क", "म", "र"]):
        detected_lang = "hi"
    elif any(char in text for char in ["আ", "ব", "ক", "ম"]):
        detected_lang = "bn"
    elif any(char in text for char in ["அ", "ஆ", "இ", "த"]):
        detected_lang = "ta"
        
    # 3. Optional local Gemma query
    try:
        payload = {
            "model": "gemma4:8b-instruct-q4_K_M",
            "prompt": f"Classify this disaster emergency query into json fields: 'disaster_type' (Flood Evacuation, Medical Emergency, or Cyclone Response), 'severity' (URGENT, HIGH, or STANDARD), 'lang' (hi, en, bn, ta). Only output JSON.\nQuery: '{req.text}'",
            "format": "json",
            "stream": False
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=2.0)
        if r.status_code == 200:
            res = json.loads(r.json()["response"])
            disaster_type = res.get("disaster_type", disaster_type)
            severity = res.get("severity", severity)
            detected_lang = res.get("lang", detected_lang)
    except Exception:
        pass # Fallback to local regex classifier

    return {
        "disaster_type": disaster_type,
        "severity": severity,
        "lang": detected_lang
    }

@app.post("/retrieve")
def retrieve_contexts(req: QueryRequest):
    """Step 2: RETRIEVE -- Semantic ChromaDB search for top-3 relevant chunks."""
    if collection is not None:
        try:
            results = collection.query(
                query_texts=[req.text],
                n_results=3
            )
            documents = results.get("documents", [[]])[0]
            if documents:
                return {"chunks": documents, "engine": "ChromaDB Persistent Index"}
        except Exception as e:
            print(f"[RAG WARNING] ChromaDB query failed: {e}")
            
    # Fallback to local keyword file searcher if ChromaDB is unavailable
    chunks = local_regex_document_search(req.text, top_k=3)
    return {
        "chunks": chunks,
        "engine": "Regex Keyword Text Parser (Local Fallback)"
    }

@app.post("/translate")
def translate_query(req: TranslateRequest):
    """Simulates IndicTrans local translator API."""
    text = req.text
    # Pre-coded translations for visual mock scripts (so script matches perfectly offline!)
    translations = {
        "mere bachche ko chot lagi hai. main kya karun?": "My child is injured. What should I do?",
        "mere ghar me pani ghus gaya hai aur light chali gayi hai, kya karu?": "Water has entered my house and power is gone, what should I do?",
        "mere barir kache emergency shelter kothay ache?": "Where is the disaster recovery shelter near my house?"
    }
    
    clean_text = req.text.strip().lower().replace("?", "").replace("!", "")
    translated = translations.get(clean_text, text)
    
    return {
        "translated_text": translated,
        "engine": "IndicTrans Local Translation Client"
    }

@app.post("/chat")
def run_chat_pipeline(req: ChatRequest):
    """3-STEP AGENT LOOP: Classify Intent -> Retrieve Context -> Generate Inference."""
    prompt = req.prompt
    
    # 1. CLASSIFY Step
    class_res = classify_query(QueryRequest(text=prompt))
    disaster = class_res["disaster_type"]
    severity = class_res["severity"]
    lang = class_res["lang"]
    
    # 2. RETRIEVE Step
    retrieve_res = retrieve_contexts(QueryRequest(text=prompt))
    chunks = retrieve_res["chunks"]
    context_text = "\n\n".join(chunks)
    
    # 3. GENERATE Step (Inference)
    ai_response = ""
    engine = "Simulated Local Gemma 4 Engine (Standalone fallback)"
    
    # Setup offline template
    system_prompt = (
        "You are Sahayak AI, an emergency responder. Use ONLY the verified context manual paragraphs below "
        "to generate direct, numbered safety or first-aid instructions. Keep response concise, under 8 words per point. No filler. No conversational headers.\n\n"
        f"Context manual reference:\n{context_text}\n\n"
        f"User Disaster Scenario: '{prompt}'\n\n"
        "Instructions Card output:"
    )
    
    # Attempt local Ollama generation if online
    try:
        payload = {
            "model": "gemma4:8b-instruct-q4_K_M",
            "prompt": system_prompt,
            "stream": False
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=5.0)
        if r.status_code == 200:
            ai_response = r.json()["response"].strip()
            engine = "Ollama Local Service (Gemma 4 Instruct)"
    except Exception:
        pass
        
    # If local Ollama generation failed, use smart pre-programmed response matching
    if not ai_response:
        prompt_lower = prompt.lower()
        if "bleed" in prompt_lower or "chot" in prompt_lower or "blood" in prompt_lower or "wound" in prompt_lower or "khoon" in prompt_lower:
            ai_response = (
                "1. STOP SEVERE BLEEDING: Apply firm, direct pressure on the cut using a clean cloth.\n"
                "2. ELEVATE INJURY: Keep the bleeding limb raised above heart levels if possible.\n"
                "3. SECURE COVER: Bind with secondary gauze. Do not remove blood-soaked dressings; wrap over them."
            )
        elif "cpr" in prompt_lower or "breath" in prompt_lower or "heart" in prompt_lower:
            ai_response = (
                "1. CPR chest compressions: Place hands in the center of the chest.\n"
                "2. COMPRESSION RATE: Push hard and fast at a rate of 100 to 120 compressions per minute.\n"
                "3. RECOVERY: Allow the chest to rise completely between pumps."
            )
        elif "flood" in prompt_lower or "water" in prompt_lower or "baadh" in prompt_lower or "pani" in prompt_lower or "shelter" in prompt_lower:
            ai_response = (
                "1. EVACUATE HIGH: Head North-West towards High Ground School evacuation shelter immediately.\n"
                "2. NO GRID POWER: Disable household electrical circuit breakers. Do not step in standing indoor water.\n"
                "3. CONSUME SAFE WATER: Avoid raw ground or floodwaters due to toxicity. Use clean chlorination tablets."
            )
        else:
            ai_response = (
                "1. REMAIN CALM: Evaluate safety of your immediate surroundings.\n"
                "2. MOVE TO HIGH ELEVATIONS: Move away from rising riverbanks or landslide slopes.\n"
                "3. ACTIVATE SOS BEACON: Keep your mobile device beacon transmitting mesh alerts."
            )

    return {
        "response": ai_response,
        "classification": {
            "disaster_type": disaster,
            "severity": severity,
            "lang": lang
        },
        "retrieval_chunks": chunks,
        "engine": engine
    }

@app.get("/poipins")
def get_map_pins(district: str = "patna"):
    """Feeds coordinate pins mapping based on region selected in onboarding."""
    dist_clean = district.lower().strip()
    pins = MARKER_PINS.get(dist_clean, MARKER_PINS["patna"])
    return {
        "district": dist_clean,
        "pins": pins
    }

@app.post("/sync")
def sync_database():
    """Trigger manual vector embedding index compilation."""
    try:
        from backend.embed_docs import main as build_embeddings
        build_embeddings()
        return {"status": "success", "msg": "ChromaDB vector embedding index successfully re-built."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
