#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re

def list_damerau_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        d[i][0] = i
    for j in range(1, n+1):
        d[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
            if (
                i > 1 and j > 1
                and seq1[i-1] == seq2[j-2]
                and seq1[i-2] == seq2[j-1]
            ):
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
    return d[m][n]

def parse_pairs(ans_str):
    pairs = []
    for tok in ans_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split()
        if len(parts) >= 2:
            verb = parts[0]
            noun = ' '.join(parts[1:])
            pairs.append((verb, noun))
    return pairs

def load_gt(gt_path):
    gt_map = {}
    tag_re = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            images = rec.get('images', [])
            if not images:
                continue
            clip_uid = os.path.basename(os.path.dirname(images[0]))

            msgs = rec.get('messages', [])
            asst = [m for m in msgs if m.get('role') == 'assistant']
            if not asst:
                continue
            content = asst[-1]['content']
            m = tag_re.search(content)
            gt_ans = m.group(1).strip() if m else ''
            gt_map[clip_uid] = gt_ans
    return gt_map

def main():
    gt_path   = "val.jsonl"       # placeholder path; replace with your actual GT file path if needed
    pred_path = "answer_val.jsonl"
    out_path  = "consensus_with_gt_normalized.json"

    gt_map = load_gt(gt_path)
    tag_re = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    results = []
    total_norm_act = 0.0
    total_norm_verb = 0.0
    total_norm_noun = 0.0
    count = 0

    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            clip_uid = rec['clip_uid']
            preds = rec['res_list']

            if clip_uid not in gt_map:
                print(f"Warning: clip_uid {clip_uid} not found in GT. Skipping.")
                continue

            gt_str = gt_map[clip_uid]
            gt_pairs = parse_pairs(gt_str)
            gt_act  = [f"{v} {n}" for v, n in gt_pairs]
            gt_verb = [v for v, n in gt_pairs]
            gt_noun = [n for v, n in gt_pairs]
            L = len(gt_act)

            d_act = []
            d_vrb = []
            d_noun = []

            for p in preds:
                m = tag_re.search(p)
                ans_str = m.group(1).strip() if m else p.strip()
                pairs = parse_pairs(ans_str)

                act_pred  = [f"{v} {n}" for v, n in pairs][:L]
                vrb_pred  = [v for v, n in pairs][:L]
                noun_pred = [n for v, n in pairs][:L]

                d_act.append(list_damerau_distance(gt_act, act_pred))
                d_vrb.append(list_damerau_distance(gt_verb, vrb_pred))
                d_noun.append(list_damerau_distance(gt_noun, noun_pred))

            best_i_act  = min(range(len(preds)), key=lambda i: d_act[i])
            best_i_vrb  = min(range(len(preds)), key=lambda i: d_vrb[i])
            best_i_noun = min(range(len(preds)), key=lambda i: d_noun[i])

            norm_act  = d_act[best_i_act] / L if L else 0.0
            norm_verb = d_vrb[best_i_vrb] / L if L else 0.0
            norm_noun = d_noun[best_i_noun] / L if L else 0.0

            results.append({
                "clip_uid": clip_uid,
                "best_by_action": {
                    "idx": best_i_act,
                    "answer": preds[best_i_act],
                    "raw_distance": d_act[best_i_act],
                    "norm_distance": round(norm_act, 4)
                },
                "best_by_verb": {
                    "idx": best_i_vrb,
                    "answer": preds[best_i_vrb],
                    "raw_distance": d_vrb[best_i_vrb],
                    "norm_distance": round(norm_verb, 4)
                },
                "best_by_noun": {
                    "idx": best_i_noun,
                    "answer": preds[best_i_noun],
                    "raw_distance": d_noun[best_i_noun],
                    "norm_distance": round(norm_noun, 4)
                }
            })

            total_norm_act  += norm_act
            total_norm_verb += norm_verb
            total_norm_noun += norm_noun
            count += 1

    with open(out_path, 'w', encoding='utf-8') as wf:
        json.dump(results, wf, ensure_ascii=False, indent=2)

    if count:
        print(f"Processed clip_uids: {count}")
        print(f"Mean normalized action distance: {total_norm_act/count:.4f}")
        print(f"Mean normalized verb distance:   {total_norm_verb/count:.4f}")
        print(f"Mean normalized noun distance:   {total_norm_noun/count:.4f}")
    else:
        print("No samples were processed.")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
