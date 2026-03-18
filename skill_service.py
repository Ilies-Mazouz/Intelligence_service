import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from fuzzywuzzy import fuzz
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="Skillsy Intelligence Service (NL)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Multilingual Model (supports Dutch)
print("Loading LLM model (MiniLM-L12)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Defined Canonical Skills (Baseline for normalization)
# This should ideally be synced with Firestore 'skillConcepts'
canonical_skills = [
    {"id": "c1", "label": "Elektrische gitaar", "rootId": "muziek", "usage": 150},
    {"id": "c2", "label": "Akoestische gitaar", "rootId": "muziek", "usage": 90},
    {"id": "c3", "label": "Piano", "rootId": "muziek", "usage": 70},
    {"id": "c4", "label": "Zang", "rootId": "muziek", "usage": 80},
    {"id": "c5", "label": "Regada (Marokkaanse dans)", "rootId": "muziek", "usage": 60},
    {"id": "c6", "label": "Fitness coaching", "rootId": "sport", "usage": 200},
    {"id": "c7", "label": "Yoga", "rootId": "sport", "usage": 120},
    {"id": "c8", "label": "Python programmeren", "rootId": "tech", "usage": 180},
    {"id": "c9", "label": "HTML & CSS", "rootId": "tech", "usage": 160},
    {"id": "c10", "label": "JavaScript / React", "rootId": "tech", "usage": 160},
    {"id": "c11", "label": "Schilderen", "rootId": "creatief", "usage": 50},
    {"id": "c12", "label": "Frans praten", "rootId": "talen", "usage": 80},
    {"id": "c13", "label": "Engels bijles", "rootId": "talen", "usage": 110},
]

labels = [s["label"] for s in canonical_skills]
embeddings = model.encode(labels)

# 3. Create FAISS index for vector search
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype('float32'))

class ResolveRequest(BaseModel):
    text: str
    locale: str = "nl"

@app.get("/")
async def root():
    return {
        "message": "Skillsy Intelligence Service is Online!",
        "endpoints": {
            "POST /resolve-skill": "Resolve and normalize skills",
            "GET /health": "Check service health"
        }
    }

@app.post("/resolve-skill")
async def resolve_skill(request: ResolveRequest):
    input_text = request.text.strip().lower()

    # 1. Exact match check (Still keep this for performance/known IDs)
    for skill in canonical_skills:
        if input_text == skill["label"].lower() or input_text == skill["id"].lower():
            return {"type": "auto_map", "match": {"concept": skill, "score": 1.0}}
            
    # Hardcoded tech redirects for precision
    tech_shortcuts = ["css", "html", "sql", "java", "php", "cpp", "js", "react", "node", "docker", "typescript"]
    if input_text in tech_shortcuts:
        input_text = f"{input_text} programming coding technology"

    # 2. Force Web-Augmented Research for every non-exact match
    import requests
    discovery_text = input_text
    is_web_augmented = False
    
    print(f"🔍 Always-on Web Discovery for: '{input_text}'")
    try:
        headers = {'User-Agent': 'SkillsyIntelligence/1.0 (contact: support@skillsy.app)'}
        
        # Phase 1: DuckDuckGo Instant Answer
        ddg_url = f"https://api.duckduckgo.com/?q={input_text}&format=json&no_html=1"
        response = requests.get(ddg_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("AbstractText"):
                discovery_text = data["AbstractText"]
                is_web_augmented = True
                print(f"🌐 DDG Result: {discovery_text[:100]}...")

        # Phase 2: Wikipedia Fallback (if DDG failed)
        if not is_web_augmented:
            print(f"🔄 DDG empty, trying Wikipedia Fallback for '{input_text}'...")
            wiki_url = f"https://nl.wikipedia.org/api/rest_v1/page/summary/{input_text.capitalize()}"
            wiki_res = requests.get(wiki_url, headers=headers, timeout=5)
            if wiki_res.status_code == 200:
                wiki_data = wiki_res.json()
                if wiki_data.get("extract"):
                    discovery_text = wiki_data["extract"]
                    is_web_augmented = True
                    print(f"📖 Wiki Result: {discovery_text[:100]}...")
            elif wiki_res.status_code == 404:
                # Try English Wikipedia if Dutch fails
                print(f"🔄 NL Wiki 404, trying EN Wiki...")
                wiki_url_en = f"https://en.wikipedia.org/api/rest_v1/page/summary/{input_text.capitalize()}"
                wiki_res_en = requests.get(wiki_url_en, headers=headers, timeout=5)
                if wiki_res_en.status_code == 200:
                    wiki_data_en = wiki_res_en.json()
                    discovery_text = wiki_data_en.get("extract", discovery_text)
                    is_web_augmented = True
                    print(f"📖 EN Wiki Result: {discovery_text[:100]}...")

    except requests.exceptions.Timeout:
        print(f"⏱️ Web search TIMEOUT voor '{input_text}'")
    except Exception as e:
        print(f"⚠️ Web search ERROR: {e}")

    # 3. Semantic Search using the ENRICHED text (from web or input)
    input_vector = model.encode([discovery_text]).astype('float32')
    distances, indices = index.search(input_vector, 3)
    
    candidates = []
        
    for i, idx in enumerate(indices[0]):
        concept = canonical_skills[idx]
        semantic_score = float(1 / (1 + distances[0][i]))
        fuzzy_score = fuzz.partial_ratio(input_text, concept["label"].lower()) / 100
        pop_score = concept["usage"] / 250
        total_score = (0.55 * semantic_score) + (0.30 * fuzzy_score) + (0.15 * pop_score)
        
        candidates.append({
            "concept": concept,
            "score": total_score
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best_match = candidates[0] if candidates else None

    # Threshold check
    if best_match and best_match["score"] >= 0.88:
        return {"type": "auto_map", "match": best_match, "isWebAugmented": is_web_augmented}
    elif best_match and best_match["score"] >= 0.75:
        return {"type": "nudge", "suggestions": candidates, "isWebAugmented": is_web_augmented}

    # 4. If no high-confidence match, use AI Categorization from Web Text
    root_categories = [
        {"id": "muziek", "label": "Muziek maken, instrumenten bespelen zoals viool of gitaar, zingen, dansen, ritme en orkest."},
        {"id": "sport", "label": "Sporten, fitness, fysieke training, coaching in sport en atletiek, beweging en gym."},
        {"id": "talen", "label": "Talen leren, vreemde talen spreken en schrijven, grammatica, vertalen en conversatie."},
        {"id": "tech", "label": "Technologie, computer programmeren, software development, websites maken, coderen en IT hardware."},
        {"id": "creatief", "label": "Creatieve kunsten, schilderen, tekenen, beeldhouwen, knutselen en handgemaakte kunstwerken."},
        {"id": "academisch", "label": "Academische vakken, bijles school, wiskunde, wetenschap, geschiedenis en studiebegeleiding."},
        {"id": "design", "label": "Grafisch design, merkidentiteit, logo's ontwerpen, user interface en digitale vormgeving."},
        {"id": "koken", "label": "Koken en bakken, culinaire vaardigheden, recepten voorbereiden en voeding in de keuken."},
        {"id": "business", "label": "Business en zaken, marketing strategie, verkoop, management, financiën en ondernemerschap."},
        {"id": "zorg", "label": "Gezondheidszorg, medische hulp, verpleging, welzijn, therapie en eerste hulp."},
        {"id": "ambacht", "label": "Handwerk en ambachten, DIY projecten, houtbewerking, reparatie en bouwen met gereedschap."},
        {"id": "fotografie", "label": "Fotografie en videografie, fotos maken met camera, film montage en visuele media."},
        {"id": "overig", "label": "Andere diverse onderwerpen die niet in een specifieke categorie passen."},
    ]
    
    root_labels = [r["label"] for r in root_categories]
    root_embeddings = model.encode(root_labels)
    root_similarities = util.cos_sim(input_vector, root_embeddings)[0]
    
    for idx, cat in enumerate(root_categories):
        f.write(f"  - {cat['id']}: {root_similarities[idx]:.4f}\n")
            
    best_root_idx = np.argmax(root_similarities)
    best_root = root_categories[best_root_idx]

    return {
        "type": "discovery",
        "proposed": {
            "label": request.text.strip().capitalize(),
            "rootId": best_root["id"],
            "rootLabel": best_root["label"].split(',')[0].strip().capitalize()
        },
        "isWebAugmented": is_web_augmented
    }

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Skillsy Intelligence is running locally."}

if __name__ == "__main__":
    # Get local IP for convenience
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n🚀 Service starting on http://{local_ip}:8000")
    print(f"👉 Use this IP in your React Native 'skillIntelligenceService.ts'")
    
    # Test print to verify labels are correctly loaded
    test_cat = "Muziek maken, instrumenten bespelen zoals viool of gitaar"
    print(f"✅ AI Labels Ready (Check: '{test_cat[:30]}...')\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
