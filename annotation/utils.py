import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
embedding_model = None
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

def call_gpt(prompt: str, temperature=0.7) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        store=True,
    )
    content = response.choices[0].message.content
    json_start = content.find("```json")
    if json_start != -1:
        json_end = content.rfind("```", json_start + 7)
        json_end = json_end if json_end != -1 else len(content)
        json_str = content[json_start + 7:json_end].strip()
        return {"reasoning": content[:json_start].strip(), "data": json.loads(json_str)}
    try:
        return {"reasoning": "", "data": json.loads(content)}
    except:
        raise ValueError(f"Could not parse JSON from response: {content[:200]}")

def batch_generate(prompts: List[str], max_workers: int = 32, save_callback=None, save_interval: int = 50, temperature: float = 0.7) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_gpt, prompt, temperature): i for i, prompt in enumerate(prompts)}
        with tqdm(total=len(prompts), desc="Generating", unit="item") as pbar:
            for future in as_completed(futures):
                try:
                    results.append((futures[future], future.result()))
                    # Incremental save callback
                    if save_callback and len(results) % save_interval == 0:
                        sorted_results = sorted(results, key=lambda x: x[0])
                        save_callback([r[1] for r in sorted_results if r[1] is not None])
                except Exception as e:
                    print(f"\nError in prompt {futures[future]}: {e}")
                    results.append((futures[future], None))
                pbar.update(1)
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results if r[1] is not None]

def filter_by_similarity(texts: List[str], threshold: float = 0.85) -> List[int]:
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
    if not texts:
        return []
    embeddings = embedding_model.encode(texts)
    similarities = cosine_similarity(embeddings)
    keep_indices = []
    for i in range(len(texts)):
        similar = False
        for j in keep_indices:
            if similarities[i][j] > threshold:
                similar = True
                break
        if not similar:
            keep_indices.append(i)
    return keep_indices

def load_json(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_json(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json_append(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json_readable(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
