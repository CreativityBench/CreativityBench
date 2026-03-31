import json, os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

def call_gpt(prompt, temperature=0.7):
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=temperature
    )
    content = resp.choices[0].message.content
    s, e = content.find("```json"), content.rfind("```")
    if s != -1 and e > s:
        return json.loads(content[s + 7:e].strip())
    return json.loads(content)

def batch_call(prompts, max_workers=16, temperature=0.7):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(call_gpt, p, temperature): i for i, p in enumerate(prompts)}
        for f in tqdm(as_completed(futures), total=len(prompts), desc="Comparing"):
            i = futures[f]
            try:
                results[i] = f.result()
            except Exception as e:
                print(f"Error {i}: {e}")
    return results

def get_embeddings(texts, batch_size=100):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model="text-embedding-3-large")
        all_embs.extend([e.embedding for e in resp.data])
    return all_embs

def load_json(path):
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
