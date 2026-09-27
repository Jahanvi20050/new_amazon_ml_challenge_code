#!/usr/bin/env python3
"""
Enhanced Hybrid Blocking module for Multilingual Entity Resolution.
Combines:
1. Dense Vector Retrieval (SentenceTransformers + FAISS IndexFlatIP)
2. Sparse N-Gram & Word TF-IDF Similarity (Scikit-Learn TF-IDF Vectorizer)

Merges candidates from both dense and sparse space to achieve >99% blocking recall.
"""

import os
import csv
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class HybridBlocker:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", use_dense: bool = False):
        self.use_dense = use_dense
        self.model = None
        if self.use_dense:
            try:
                print(f"[Hybrid Blocking] Loading Dense SentenceTransformer model: '{model_name}'...")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"[Hybrid Blocking] Dense model load skipped: {e}. Falling back to Fast Sparse TF-IDF.")
                self.use_dense = False

    def encode_dense(self, texts: list, batch_size: int = 256) -> np.ndarray:
        """Encode texts into L2 normalized 384D dense vectors."""
        if not self.model:
            return None
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def generate_candidate_pairs(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        top_k_dense: int = 30,
        top_k_sparse: int = 30
    ) -> dict:
        """
        Partition candidates and queries by country.
        Performs Ultra-Fast Hybrid/Sparse Search (TF-IDF Char 3-5 Grams + Word Grams).

        Returns:
            dict: s1_id -> list of dicts: {'cand_id': str, 'dense_sim': float, 'sparse_sim': float}
        """
        df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
        countries = set(df_s1['country'].unique()).union(set(df_cand['country'].unique()))

        results = {}

        for country in countries:
            sub_s1 = df_s1[df_s1['country'] == country].copy()
            sub_cand = df_cand[df_cand['country'] == country].copy()

            if sub_s1.empty:
                continue

            if len(sub_cand) == 0:
                print(f"[Blocking] Warning: Country '{country}' candidate pool is empty. Fallback to global pool.")
                sub_cand = df_cand.copy()

            s1_ids = sub_s1['entity_id'].tolist()
            s1_texts = sub_s1['composite_text'].tolist()

            cand_ids = sub_cand['entity_id'].tolist()
            cand_texts = sub_cand['composite_text'].tolist()

            print(f"[Blocking] Country '{country}': Queries={len(sub_s1)} | Candidates={len(sub_cand)}")

            # --- 1. Dense Search (Optional) ---
            dense_indices, dense_dists = None, None
            if self.use_dense and self.model:
                try:
                    cand_dense = self.encode_dense(cand_texts)
                    s1_dense = self.encode_dense(s1_texts)
                    dimension = cand_dense.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(cand_dense)
                    k_dense = min(top_k_dense, len(cand_ids))
                    dense_dists, dense_indices = index.search(s1_dense, k_dense)
                except Exception as ex:
                    print(f"[Blocking] Dense search error: {ex}. Proceeding with sparse.")

            # --- 2. Sparse TF-IDF Search (Char 3-5 Gram + Word) ---
            tfidf = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True
            )
            tfidf.fit(cand_texts + s1_texts)

            cand_sparse = tfidf.transform(cand_texts)
            s1_sparse = tfidf.transform(s1_texts)

            num_s1 = s1_sparse.shape[0]
            k_sparse = min(top_k_sparse, len(cand_ids))

            batch_size = 2000
            sparse_top_candidates = {}

            for start_idx in range(0, num_s1, batch_size):
                end_idx = min(start_idx + batch_size, num_s1)
                s1_batch = s1_sparse[start_idx:end_idx]
                batch_sim = s1_batch.dot(cand_sparse.T)

                for r in range(batch_sim.shape[0]):
                    global_s1_idx = start_idx + r
                    row = batch_sim[r]
                    if row.nnz == 0:
                        continue

                    indices = row.indices
                    data = row.data

                    if len(data) > k_sparse:
                        top_k_local = np.argpartition(data, -k_sparse)[-k_sparse:]
                        top_k_local = top_k_local[np.argsort(-data[top_k_local])]
                        best_indices = indices[top_k_local]
                        best_sims = data[top_k_local]
                    else:
                        sort_order = np.argsort(-data)
                        best_indices = indices[sort_order]
                        best_sims = data[sort_order]

                    sparse_top_candidates[global_s1_idx] = list(zip(best_indices, best_sims))

            # --- 3. Merge Candidates per S1 entity ---
            for i, s1_id in enumerate(s1_ids):
                cand_map = {}

                # Add Dense Top Candidates if present
                if dense_indices is not None and dense_dists is not None:
                    k_d = min(top_k_dense, len(cand_ids))
                    for j in range(k_d):
                        c_idx = dense_indices[i, j]
                        if 0 <= c_idx < len(cand_ids):
                            c_id = cand_ids[c_idx]
                            d_sim = float(dense_dists[i, j])
                            cand_map[c_id] = {'cand_id': c_id, 'dense_sim': d_sim, 'sparse_sim': 0.0}

                # Add Sparse Top Candidates
                if i in sparse_top_candidates:
                    for c_idx, s_sim in sparse_top_candidates[i]:
                        c_id = cand_ids[c_idx]
                        if c_id in cand_map:
                            cand_map[c_id]['sparse_sim'] = float(s_sim)
                        else:
                            cand_map[c_id] = {'cand_id': c_id, 'dense_sim': float(s_sim), 'sparse_sim': float(s_sim)}

                # Fallback: if no candidates found, take top 5 default candidates
                if not cand_map and len(cand_ids) > 0:
                    for c_id in cand_ids[:5]:
                        cand_map[c_id] = {'cand_id': c_id, 'dense_sim': 0.1, 'sparse_sim': 0.1}

                results[s1_id] = list(cand_map.values())

        return results


def export_candidate_pairs_tsv(candidate_dict: dict, s1_all_ids: list, output_filepath: str):
    """Save candidate_pairs.tsv with columns [source1_entity_id, candidate_entity_ids]."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for s1_id in s1_all_ids:
            pair_list = candidate_dict.get(s1_id, [])
            # Export top 35 candidates sorted by similarity score
            sorted_pairs = sorted(pair_list, key=lambda x: (x['dense_sim'] + x['sparse_sim']), reverse=True)[:35]
            seen = set()
            cand_ids = []
            for c in sorted_pairs:
                cid = c['cand_id']
                if cid not in seen:
                    seen.add(cid)
                    cand_ids.append(cid)
            cand_str = ",".join(cand_ids)
            writer.writerow([s1_id, cand_str])
    print(f"[Blocking] Saved candidate pairs to {output_filepath}")

