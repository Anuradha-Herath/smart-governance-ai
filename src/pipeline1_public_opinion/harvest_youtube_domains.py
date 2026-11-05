"""Multi-Domain YouTube Public Opinion Harvester for Pipeline 1.

Harvests and standardizes citizen reactions across all 6 research domains defined in
`configs/manifest_youtube_policy_cases.yaml`:
1. Electricity and Fuel
2. Agriculture
3. Customs, Imports and Exports
4. Food Security and Food Prices
5. Government Finance and Taxation
6. Healthcare and Public Health

Outputs:
- Datasets/22_sl_multidomain_public_opinion/sl_multidomain_public_opinion.parquet
- Datasets/22_sl_multidomain_public_opinion/sl_multidomain_public_opinion.csv
- data/processed/p1_multidomain_comments.parquet
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.request
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline1_public_opinion.collect_decision_comments import (
    CHANNEL_EXCLUDE_RE,
    HEADERS,
    Tutorial_EXCLUDE_RE,
    _find_dict_values,
    classify_tariff_sentiment_and_stance,
    detect_language_modality,
    search_youtube_videos,
)
from src.pipeline1_public_opinion.predict import get_sentiment_model

MANIFEST_PATH = PROJECT_ROOT / "configs" / "manifest_youtube_policy_cases.yaml"
EXISTING_ELEC_PQ = PROJECT_ROOT / "data" / "processed" / "p1_decision_linked_comments.parquet"
DATASET_OUT_DIR = PROJECT_ROOT.parent / "Datasets" / "22_sl_multidomain_public_opinion"
PROCESSED_OUT_PQ = PROJECT_ROOT / "data" / "processed" / "p1_multidomain_comments.parquet"
REGISTER_CSV = PROJECT_ROOT.parent / "Datasets" / "dataset_register.csv"


def assign_temporal_window(pub_date_str: str, enactment_date_str: str) -> str:
    """Classify broadcast date into T_pre, T_event (0 to +30 days), or T_post (>30 days)."""
    try:
        pub_dt = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
        enact_dt = datetime.strptime(enactment_date_str[:10], "%Y-%m-%d")
        delta_days = (pub_dt - enact_dt).days
        if delta_days < 0:
            return "T_pre (Anticipation)"
        if delta_days <= 30:
            return "T_event (Enactment Shock)"
        return "T_post (Adaptation & Impact)"
    except Exception:
        return "T_event (Enactment Shock)"


def classify_aspect_category(text: str, domain_id: int) -> str:
    """Assign primary policy grievance/reaction aspect category."""
    t = text.lower()
    if re.search(
        r"(?:හොර|හොරු|වංචා|දූෂණ|කොමිස්|මන්ත්‍රී|ඇමති|දේශපාලන|පාර්ලිමේන්තු|බෝනස්|වැටුප්|"
        r"corrupt|hora|horu|politician|minister|commission|waste|mafia)",
        t,
    ):
        return "Governance & Corruption"
    if re.search(
        r"(?:හිඟ|නැහැ|නෑ|පෝලිම්|පෝලිමේ|කප්පාදු|කපනවා|නැතිව|බාල|ලැබෙන්නේ නෑ|"
        r"shortage|queue|blackout|power cut|unavailable|scarcity)",
        t,
    ):
        return "Supply & Shortage"
    if re.search(
        r"(?:ගොවි|වගා|අස්වැන්න|කුඹුරු|ව්‍යාපාර|කර්මාන්ත|රැකියා|පඩි|වැටුප්|බේකරි|හෝටල්|"
        r"farmer|crop|harvest|business|sme|industry|job|salary|export|import)",
        t,
    ):
        return "Livelihood & Production"
    if re.search(
        r"(?:මිල|බිල|ගාස්තු|බදු|වැට්|වැඩි|දරාගන්න|ජීවත්|බඩගින්න|කන්න|සල්ලි|රුපියල්|"
        r"price|bill|tax|vat|tariff|cost|afford|expensive|wadi|bila)",
        t,
    ):
        return "Cost of Living & Pricing"
    return {
        1: "Cost of Living & Pricing",
        2: "Livelihood & Production",
        3: "Supply & Shortage",
        4: "Cost of Living & Pricing",
        5: "Cost of Living & Pricing",
        6: "Supply & Shortage",
    }.get(domain_id, "General Policy Reaction")


def parse_like_count(raw_val: Any) -> int:
    if not raw_val:
        return 0
    s = str(raw_val).strip().upper()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KM])?", s)
    if not m:
        return 0
    val = float(m.group(1))
    if m.group(2) == "K":
        val *= 1000
    elif m.group(2) == "M":
        val *= 1000000
    return int(val)


def fetch_video_comments_with_likes(video_id: str, max_pages: int = 5) -> tuple[str, list[dict[str, Any]]]:
    """Fetch YouTube comments along with publish date, relative time, and engagement likes."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    except Exception:
        return "", []

    key_m = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
    date_m = re.search(r'"publishDate":"([^"]+)"', html)
    init_m = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
    if not key_m or not init_m:
        return "", []

    api_key = key_m.group(1)
    pub_date = date_m.group(1) if date_m else ""
    init_data = json.loads(init_m.group(1))

    tokens = []
    for item_sec in _find_dict_values(init_data, "itemSectionRenderer"):
        if item_sec.get("sectionIdentifier") == "comment-item-section" or "comments" in str(item_sec.get("targetId", "")):
            for tok in _find_dict_values(item_sec, "token"):
                tokens.append(tok)
    if not tokens:
        for cmd in _find_dict_values(init_data, "continuationCommand"):
            tok = cmd.get("token", "")
            if tok and len(tok) > 40:
                tokens.append(tok)

    comments: list[dict[str, Any]] = []
    visited: set[str] = set()
    queue = tokens[:2]
    pages = 0

    while queue and pages < max_pages:
        tok = queue.pop(0)
        if tok in visited:
            continue
        visited.add(tok)
        pages += 1

        api_url = f"https://www.youtube.com/youtubei/v1/next?key={api_key}"
        payload = json.dumps({
            "context": {
                "client": {
                    "hl": "en",
                    "gl": "LK",
                    "clientName": "WEB",
                    "clientVersion": "2.20240726.00.00",
                }
            },
            "continuation": tok,
        }).encode("utf-8")
        creq = urllib.request.Request(
            api_url,
            data=payload,
            headers={**HEADERS, "Content-Type": "application/json"},
        )
        try:
            cresp = json.loads(urllib.request.urlopen(creq, timeout=15).read().decode("utf-8", errors="ignore"))
        except Exception:
            continue

        for cep in _find_dict_values(cresp, "commentEntityPayload"):
            props = cep.get("properties", {})
            toolbar = cep.get("toolbar", {})
            cid = props.get("commentId", "")
            content = props.get("content", {}).get("content", "").strip()
            pub_time = props.get("publishedTime", "")
            likes = parse_like_count(toolbar.get("likeCountNotliked") or toolbar.get("likeCountA11y"))
            if cid and content and len(content) >= 4:
                comments.append({
                    "comment_id": f"YT_{cid}",
                    "text_raw": content,
                    "relative_time": pub_time,
                    "likes": likes,
                })

        for cr in _find_dict_values(cresp, "commentRenderer"):
            cid = cr.get("commentId", "")
            content = "".join(r.get("text", "") for r in cr.get("contentText", {}).get("runs", [])).strip()
            pub_time = "".join(r.get("text", "") for r in cr.get("publishedTimeText", {}).get("runs", []))
            likes = parse_like_count(cr.get("voteCount", {}).get("simpleText", "0"))
            if cid and content and len(content) >= 4:
                comments.append({
                    "comment_id": f"YT_{cid}",
                    "text_raw": content,
                    "relative_time": pub_time,
                    "likes": likes,
                })

        for cmd in _find_dict_values(cresp, "continuationCommand"):
            ntok = cmd.get("token", "")
            if ntok and ntok not in visited and len(ntok) > 40:
                queue.append(ntok)

    dedup = {c["comment_id"]: c for c in comments}
    return pub_date, list(dedup.values())


def harvest_all_domains() -> pd.DataFrame:
    """Harvest YouTube comments across all 6 domains and build unified dataset."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    sent_clf = get_sentiment_model()
    records: list[dict[str, Any]] = []

    # 1. Load existing verified Electricity Tariff comments (DEC_2022_ELEC_01, DEC_2023_ELEC_01, DEC_2024_ELEC_01)
    existing_elec_cases = {"DEC_2022_ELEC_01", "DEC_2023_ELEC_01", "DEC_2024_ELEC_01"}
    policy_lookup: dict[str, dict[str, Any]] = {}
    for dom in manifest["domains"]:
        for pol in dom["policies"]:
            pol["domain_id"] = dom["domain_id"]
            pol["domain_name"] = dom["domain_name"]
            policy_lookup[pol["policy_case_id"]] = pol

    # 1. Load existing harvested multidomain dataset or Electricity Tariff comments
    existing_videos: set[str] = set()
    domain_counts: dict[int, int] = {}
    if PROCESSED_OUT_PQ.exists():
        prev_df = pd.read_parquet(PROCESSED_OUT_PQ)
        for rec in prev_df.to_dict(orient="records"):
            pol = policy_lookup.get(str(rec["policy_case_id"]))
            if pol:
                rec["temporal_window"] = assign_temporal_window(str(rec["post_publish_date"]), pol["enactment_date"])
                rec["legal_anchor"] = pol["legal_anchor"]
            records.append(rec)
            existing_videos.add(str(rec["video_id"]))
            d_id = int(rec["domain_id"])
            domain_counts[d_id] = domain_counts.get(d_id, 0) + 1
        print(f"Loaded {len(prev_df):,} previously harvested comments across {len(existing_videos)} broadcasts.")
    elif EXISTING_ELEC_PQ.exists():
        elec_df = pd.read_parquet(EXISTING_ELEC_PQ)
        print(f"Loaded {len(elec_df)} existing verified electricity tariff comments into Domain 1.")
        for _, row in elec_df.iterrows():
            cid = str(row["decision_id"])
            pol = policy_lookup.get(cid, policy_lookup["DEC_2023_ELEC_01"])
            txt = str(row["text"])
            pub_date = str(row.get("video_publish_date", pol["enactment_date"]))
            vid = str(row["video_id"])
            existing_videos.add(vid)
            records.append({
                "comment_id": str(row["comment_id"]),
                "domain_id": 1,
                "domain_name": "Electricity and Fuel",
                "sub_focus": pol["sub_focus"],
                "policy_case_id": cid,
                "legal_anchor": pol["legal_anchor"],
                "text_raw": txt,
                "text_clean": re.sub(r"\s+", " ", txt).strip(),
                "language": str(row["language"]),
                "source_platform": "YouTube",
                "video_id": vid,
                "video_url": str(row["video_url"]),
                "channel": str(row["channel"]),
                "video_title": str(row["video_title"]),
                "post_publish_date": pub_date,
                "temporal_window": assign_temporal_window(pub_date, pol["enactment_date"]),
                "comment_relative_time": "archived",
                "sentiment_label": str(row["sentiment_label"]),
                "stance_label": str(row["stance_label"]),
                "aspect_category": classify_aspect_category(txt, 1),
                "engagement_likes": 0,
                "annotation_method": "Hybrid_Lexical_LogReg_Silver",
            })

    # 2. Harvest all policies needing expansion (target >= 400 comments per domain)
    for dom in manifest["domains"]:
        dom_id = dom["domain_id"]
        dom_name = dom["domain_name"]
        if domain_counts.get(dom_id, 0) >= 450:
            print(f"Skipping Domain {dom_id} ({dom_name}) — already has {domain_counts[dom_id]} comments.")
            continue
        for pol in dom["policies"]:
            case_id = pol["policy_case_id"]
            if case_id in existing_elec_cases and EXISTING_ELEC_PQ.exists():
                continue

            print(f"\n--- Harvesting Domain {dom_id}: {dom_name} [{case_id}] ---")
            cand_videos: dict[str, dict[str, str]] = {}
            for q in pol["search_queries"]:
                for v in search_youtube_videos(q, max_results=12):
                    if v["video_id"] not in existing_videos:
                        cand_videos[v["video_id"]] = v

            start_d, end_d = pol["date_range"]
            kws = [k.lower() for k in pol["title_keywords"]]
            direction = pol.get("direction", "hike_or_restriction")
            # Map direction to classify_tariff_sentiment_and_stance polarity convention
            ref_dec_id = "DEC_2023_ELEC_01" if direction == "hike_or_restriction" else "DEC_2024_ELEC_01"

            for vid, vmeta in cand_videos.items():
                title = vmeta["title"]
                channel = vmeta["channel"]
                if Tutorial_EXCLUDE_RE.search(title) or CHANNEL_EXCLUDE_RE.search(channel):
                    continue
                if not any(k in title.lower() for k in kws):
                    continue

                pub_date, comms = fetch_video_comments_with_likes(vid, max_pages=10)
                if not pub_date or not comms:
                    continue
                date_part = pub_date[:10]
                if not (start_d <= date_part <= end_d):
                    continue

                print(f"  [{case_id}] {date_part} | {vid} | {channel[:18]} | {title[:48]} -> {len(comms)} comments")
                clean_texts = [re.sub(r"\s+", " ", c["text_raw"]).strip() for c in comms]
                model_preds = sent_clf.predict(clean_texts)
                t_win = assign_temporal_window(date_part, pol["enactment_date"])

                for c, t_clean, m_sent in zip(comms, clean_texts, model_preds):
                    # Exclude Tamil script comments per research scope (Sinhala, Singlish, English only)
                    if len(t_clean) < 4 or re.search(r"[\u0B80-\u0BFF]", t_clean):
                        continue
                    lang = detect_language_modality(t_clean)
                    sent, stance = classify_tariff_sentiment_and_stance(t_clean, ref_dec_id, str(m_sent))
                    records.append({
                        "comment_id": c["comment_id"],
                        "domain_id": dom_id,
                        "domain_name": dom_name,
                        "sub_focus": pol["sub_focus"],
                        "policy_case_id": case_id,
                        "legal_anchor": pol["legal_anchor"],
                        "text_raw": c["text_raw"],
                        "text_clean": t_clean,
                        "language": lang,
                        "source_platform": "YouTube",
                        "video_id": vid,
                        "video_url": f"https://www.youtube.com/watch?v={vid}",
                        "channel": channel,
                        "video_title": title,
                        "post_publish_date": date_part,
                        "temporal_window": t_win,
                        "comment_relative_time": c["relative_time"],
                        "sentiment_label": sent,
                        "stance_label": stance,
                        "aspect_category": classify_aspect_category(t_clean, dom_id),
                        "engagement_likes": int(c["likes"]),
                        "annotation_method": "Hybrid_Lexical_LogReg_Silver",
                    })

    df = pd.DataFrame(records)
    # Enforce scope: filter out any Tamil script comments across entire dataset
    df = df[~df["text_clean"].astype(str).str.contains(r"[\u0B80-\u0BFF]", regex=True)]
    df = df.drop_duplicates(subset=["comment_id", "policy_case_id"]).reset_index(drop=True)

    DATASET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUT_PQ.parent.mkdir(parents=True, exist_ok=True)

    pq_path = DATASET_OUT_DIR / "sl_multidomain_public_opinion.parquet"
    csv_path = DATASET_OUT_DIR / "sl_multidomain_public_opinion.csv"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(PROCESSED_OUT_PQ, index=False)

    # Export 600-row Stratified Gold Verification Sheet (up to 100 comments per domain)
    gold_samples = []
    for d_id in sorted(df["domain_id"].unique()):
        sub = df[df["domain_id"] == d_id].copy()
        sub["char_len"] = sub["text_clean"].str.len()
        sub = sub[sub["char_len"] >= 20].sort_values(by=["engagement_likes", "char_len"], ascending=[False, False])
        sampled = sub.head(100).copy()
        sampled["human_verified_sentiment"] = sampled["sentiment_label"]
        sampled["human_verified_stance"] = sampled["stance_label"]
        sampled["annotator_notes"] = ""
        gold_samples.append(sampled.drop(columns=["char_len"]))
    if gold_samples:
        gold_df = pd.concat(gold_samples, ignore_index=True)
        gold_path = DATASET_OUT_DIR / "gold_annotation_sample_600.csv"
        gold_df.to_csv(gold_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(gold_df)}-row Stratified Gold Standard Verification Sheet to: {gold_path}")

    # Register in dataset_register.csv
    if REGISTER_CSV.exists():
        try:
            reg_df = pd.read_csv(REGISTER_CSV)
            reg_df = reg_df[reg_df["dataset_id"] != "22_sl_multidomain_public_opinion"]
            new_row = {
                "dataset_id": "22_sl_multidomain_public_opinion",
                "title": "Sri Lanka 6-Domain Policy Reaction YouTube Corpus (2021-2024)",
                "github_repo": "Local Research Harvest (configs/manifest_youtube_policy_cases.yaml)",
                "official_source": "YouTube Official Sri Lankan News Broadcasts (Ada Derana, Hiru, Sirasa, Newsfirst)",
                "license": "Academic Research Corpus (Anonymized Public Broadcast Comments)",
                "retrieval_date": "2026-11-04",
                "languages": "si, singlish, en, si_en_codemixed",
                "temporal_coverage": "2021-2024",
                "docs_count": int(df["video_id"].nunique()),
                "chunks_count": len(df),
                "local_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2),
                "local_path": "Datasets/22_sl_multidomain_public_opinion",
                "pipeline_fit": "Pipeline 1 (Multi-Domain Citizen Sentiment, Stance, Aspect & Temporal Reaction)",
            }
            reg_df = pd.concat([reg_df, pd.DataFrame([new_row])], ignore_index=True)
            reg_df.to_csv(REGISTER_CSV, index=False)
        except Exception as exc:
            print(f"Warning: could not update dataset register: {exc}")

    print("\n=====================================================================")
    print(f"COMPLETED 6-DOMAIN HARVEST: {len(df):,} total comments across {df['video_id'].nunique()} broadcasts")
    print("=====================================================================")
    print("\n1. Domain & Policy Breakdown:")
    summary = (
        df.groupby(["domain_id", "domain_name", "policy_case_id"])
        .agg(broadcasts=("video_id", "nunique"), comments=("comment_id", "count"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n2. Temporal Window Distribution:")
    print(df.groupby(["domain_id", "temporal_window"]).size().unstack(fill_value=0))

    print("\n3. Stance Breakdown by Domain:")
    print(df.groupby(["domain_name", "stance_label"]).size().unstack(fill_value=0))
    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    harvest_all_domains()
