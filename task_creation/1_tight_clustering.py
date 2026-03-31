import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
from utils import get_embeddings, load_json, save_json

INPUT = "../annotation/outputs/8_entities_complete.json"
OUTPUT = "outputs"

# Hard upper bound on cluster diameter (cosine distance).
# The knee-detection auto-threshold is used first; this cap applies on top of it
# so you can prevent excessively loose clusters. Set to None to disable.
MAX_DIAMETER = 0.35

LEVEL_LABELS = ["Normal 0", "Emergency 1", "Emergency 2", "Emergency 3", "Emergency 4", "Emergency 5"]

# Process given scenarios first
SCENARIOS = [
    "kitchen", "living_room", "bedroom", "bathroom", "garage", "home_office", "dining_room", "garden"
]

def parse_level(level_str):
    for lbl in LEVEL_LABELS:
        if lbl in str(level_str):
            return lbl
    return "Unknown"

def cluster_stats(scenario, scen_clusters, aff_lookup, entity_lookup):
    sizes = sorted([len(v) for v in scen_clusters.values()])
    level_counts = {l: 0 for l in LEVEL_LABELS}
    per_cluster = []
    for cid, uids in sorted(scen_clusters.items(), key=lambda x: -len(x[1])):
        lc = {l: 0 for l in LEVEL_LABELS}
        samples = []
        for uid in uids:
            aff = aff_lookup[uid]["affordance"]
            lbl = parse_level(aff.get("level", ""))
            lc[lbl] = lc.get(lbl, 0) + 1
            level_counts[lbl] = level_counts.get(lbl, 0) + 1
            samples.append(aff.get("affordance", ""))
        per_cluster.append({"id": cid, "size": len(uids), "level_counts": lc, "sample_affordances": samples[:3]})
    n_entities = sum(1 for v in entity_lookup.values() if v["scenario"] == scenario)
    
    # Count clusters by size: {size: count}
    size_distribution = {}
    for size in sizes:
        size_distribution[size] = size_distribution.get(size, 0) + 1
    
    return {
        "n_entities": n_entities,
        "n_affordances": sum(sizes),
        "n_clusters": len(sizes),
        "cluster_sizes_sorted": sizes,
        "size_distribution": size_distribution,  # {1: count, 2: count, ...}
        "size_range_counts": {"1": sizes.count(1), "2-5": sum(1 for s in sizes if 2 <= s <= 5),
                              "6-10": sum(1 for s in sizes if 6 <= s <= 10), ">10": sum(1 for s in sizes if s > 10)},
        "level_counts": level_counts,
        "per_cluster": per_cluster,
    }


def tight_cluster(embs):
    """Complete-linkage hierarchical clustering with a self-adaptive diameter threshold.

    Threshold T is chosen via knee detection on the normalized merge-distance curve:
    - Normalize merge indices and distances to [0, 1].
    - The "knee" is the point with the maximum distance below the diagonal (x - y peak).
      This is the elbow where cheap within-cluster merges give way to expensive
      between-cluster merges.
    - An optional MAX_DIAMETER cap ensures clusters never exceed a fixed cosine distance.

    Every cluster produced has a worst-case pairwise cosine distance ≤ T.
    """
    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)

    # Pairwise cosine distances → condensed form for scipy linkage
    D = cosine_distances(embs)
    condensed = D[np.triu_indices(n, k=1)]

    # Complete linkage: merge distance = max pairwise distance in merged cluster
    Z = linkage(condensed, method="complete")
    merge_dists = Z[:, 2]

    # Knee detection: normalize curve and find point furthest below the diagonal.
    # merge_dists is monotonically non-decreasing, so the curve is concave-shaped.
    # Points far below y=x are in the "cheap merge" region; the peak marks the elbow.
    m = len(merge_dists)
    if m >= 2:
        x = np.arange(m) / (m - 1)                                          # 0 → 1
        y = (merge_dists - merge_dists[0]) / max(merge_dists[-1] - merge_dists[0], 1e-12)
        T = merge_dists[int(np.argmax(x - y))]
    else:
        T = merge_dists[0] * 0.99

    # Optional hard cap: never allow a cluster diameter larger than MAX_DIAMETER
    if MAX_DIAMETER is not None:
        T = min(T, MAX_DIAMETER)

    # fcluster with criterion='distance' guarantees cluster diameter ≤ T
    return fcluster(Z, T, criterion="distance") - 1   # 0-indexed


def visualize(embs, labels, scenario, k):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(embs)
    var = pca.explained_variance_ratio_ * 100
    # Golden-ratio hue spacing: adjacent cluster IDs get maximally different hues,
    # works well for any k (including thousands of clusters).
    phi = 0.618033988749895
    colors = np.array([plt.cm.hsv((i * phi) % 1.0) for i in range(k)])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(pts[:, 0], pts[:, 1], c=colors[labels], alpha=0.6, s=8, linewidths=0)
    ax.set_title(f"{scenario}  —  k={k} tight clusters  (PCA 2D, complete-linkage)", fontsize=13)
    ax.set_xlabel(f"PC1  ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var[1]:.1f}% var)")
    plt.tight_layout()
    path = f"{OUTPUT}/embedding_figures/{scenario}_tight_clusters.png"
    os.makedirs(f"{OUTPUT}/embedding_figures", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot → {path}")


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    entities = load_json(INPUT)

    # Lookup tables: always rebuilt from source
    aff_lookup, entity_lookup = {}, {}
    uids, texts, uid_to_idx = [], [], {}
    for ent in entities:
        e_uid = f"{ent['scenario']}+{ent['entity_name']}"
        entity_lookup[e_uid] = {"entity_name": ent["entity_name"], "scenario": ent["scenario"], "parts": ent["parts"]}
        for part in ent["parts"]:
            for i, aff in enumerate(part["functional_affordances"]):
                uid = f"{e_uid}::{part['part_name']}::{i}"
                aff_lookup[uid] = {"entity_uid": e_uid, "part_name": part["part_name"],
                                   "scenario": ent["scenario"], "affordance": aff}
                uid_to_idx[uid] = len(uids)
                uids.append(uid)
                texts.append(aff["affordance"])
    save_json(aff_lookup, f"{OUTPUT}/1_affordance_lookup.json")
    save_json(entity_lookup, f"{OUTPUT}/1_entity_lookup.json")
    print(f"Entities: {len(entity_lookup)}, Affordances: {len(uids)}")

    # Embeddings: load cache, compute only missing uids, save immediately
    emb_path = f"{OUTPUT}/1_embeddings.json"
    emb_dict = load_json(emb_path) if os.path.exists(emb_path) else {}
    missing = [u for u in uids if u not in emb_dict]
    if missing:
        print(f"Embedding: {len(missing)} new ({len(emb_dict)} cached)...")
        for u, emb in zip(missing, get_embeddings([texts[uid_to_idx[u]] for u in missing])):
            emb_dict[u] = emb
        save_json(emb_dict, emb_path)
    else:
        print(f"Embeddings: all {len(emb_dict)} loaded from cache.")
    emb_array = np.array([emb_dict[u] for u in uids])

    # Clusters, centroids, stats: load cache, skip completed scenarios, save after each
    clusters  = load_json(f"{OUTPUT}/1_clusters.json")  if os.path.exists(f"{OUTPUT}/1_clusters.json")  else {}
    centroids = load_json(f"{OUTPUT}/1_centroids.json") if os.path.exists(f"{OUTPUT}/1_centroids.json") else {}
    stats     = load_json(f"{OUTPUT}/1_stats.json")     if os.path.exists(f"{OUTPUT}/1_stats.json")     else {}

    scenarios = list({v["scenario"] for v in aff_lookup.values()})
    scenarios = [s for s in scenarios if s in SCENARIOS]
    print(f"Processing {len(scenarios)} scenarios...")
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        scen_uids  = [u for u in uids if aff_lookup[u]["scenario"] == scenario]
        scen_idx   = [uid_to_idx[u] for u in scen_uids]
        scen_embs  = emb_array[scen_idx]
        scen_texts = [texts[i] for i in scen_idx]

        if scenario not in clusters:
            labels = tight_cluster(scen_embs)
            scen_clusters = {}
            for uid, label in zip(scen_uids, labels):
                scen_clusters.setdefault(str(label), []).append(uid)

            local_idx = {uid: i for i, uid in enumerate(scen_uids)}
            scen_centroids = {
                cid: scen_embs[[local_idx[u] for u in cluster_uids]].mean(axis=0).tolist()
                for cid, cluster_uids in scen_clusters.items()
            }

            clusters[scenario]  = scen_clusters
            centroids[scenario] = scen_centroids
            save_json(clusters,  f"{OUTPUT}/1_clusters.json")
            save_json(centroids, f"{OUTPUT}/1_centroids.json")

            k = len(scen_clusters)
            visualize(scen_embs, labels, scenario, k)
            print(f"  {scenario}: {len(scen_uids)} affordances → {k} tight clusters (auto-threshold)")
        else:
            print(f"  {scenario}: clusters loaded from cache (k={len(clusters[scenario])})")

        if scenario not in stats:
            stats[scenario] = cluster_stats(scenario, clusters[scenario], aff_lookup, entity_lookup)
            save_json(stats, f"{OUTPUT}/1_stats.json")
            print(f"  {scenario}: stats saved.")

    print("Done!")


if __name__ == "__main__":
    main()
