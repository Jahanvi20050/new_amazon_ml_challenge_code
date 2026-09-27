#!/usr/bin/env python3
"""
Submission validation script for Multilingual Entity Resolution Hackathon.
Validates output/matching_results.tsv and output/candidate_pairs.tsv format and integrity.
"""

import sys
import os
import csv

def validate_candidate_pairs(filepath: str) -> bool:
    if not os.path.exists(filepath):
        print(f"[ERROR] Candidate pairs file missing: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if not header or len(header) < 2 or header[0] != 'source1_entity_id' or header[1] != 'candidate_entity_ids':
            print(f"[ERROR] Invalid header in candidate pairs file. Expected ['source1_entity_id', 'candidate_entity_ids'], got {header}")
            return False

        seen_ids = set()
        count = 0
        for i, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                print(f"[ERROR] Line {i}: Empty source1_entity_id")
                return False
            if s1_id in seen_ids:
                print(f"[ERROR] Line {i}: Duplicate source1_entity_id '{s1_id}'")
                return False
            seen_ids.add(s1_id)
            c_ids = row[1].strip() if len(row) > 1 else ""
            c_list = [c.strip() for c in c_ids.split(',') if c.strip()]
            if len(c_list) > 35:
                print(f"[WARNING] Line {i}: Candidate count {len(c_list)} exceeds expected maximum of 35")
            count += 1

    print(f"[OK] Candidate pairs file validated successfully ({count} records).")
    return True

def validate_matching_results(filepath: str) -> bool:
    if not os.path.exists(filepath):
        print(f"[ERROR] Matching results file missing: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if not header or len(header) < 2 or header[0] != 'source1_entity_id' or header[1] != 'matched_entity_ids':
            print(f"[ERROR] Invalid header in matching results file. Expected ['source1_entity_id', 'matched_entity_ids'], got {header}")
            return False

        seen_ids = set()
        count = 0
        singletons = 0
        for i, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                print(f"[ERROR] Line {i}: Empty source1_entity_id")
                return False
            if s1_id in seen_ids:
                print(f"[ERROR] Line {i}: Duplicate source1_entity_id '{s1_id}'")
                return False
            seen_ids.add(s1_id)
            m_ids = row[1].strip() if len(row) > 1 else ""
            if not m_ids:
                singletons += 1
            count += 1

    print(f"[OK] Matching results file validated successfully ({count} records, {singletons} singletons/empty predictions).")
    return True

def main():
    candidate_path = os.path.join("output", "candidate_pairs.tsv")
    matching_path = os.path.join("output", "matching_results.tsv")
    matching_entities_path = os.path.join("output", "matching_entities.tsv")
    root_matching_entities_path = "matching_entities.tsv"

    valid_candidates = validate_candidate_pairs(candidate_path)
    
    target_matching = matching_path if os.path.exists(matching_path) else (
        matching_entities_path if os.path.exists(matching_entities_path) else root_matching_entities_path
    )
    valid_matching = validate_matching_results(target_matching)

    if os.path.exists(matching_entities_path):
        validate_matching_results(matching_entities_path)
    if os.path.exists(root_matching_entities_path):
        validate_matching_results(root_matching_entities_path)

    if valid_candidates and valid_matching:
        print("[SUCCESS] All submission files passed validation!")
        sys.exit(0)
    else:
        print("[FAILURE] Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
