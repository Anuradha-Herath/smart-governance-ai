"""Collect and structure real decision-linked citizen comments from Sri Lankan news broadcasts.

Targets the three electricity tariff decisions evaluated in the platform:
- DEC_2022_ELEC_01: August 2022 Initial 75% Electricity Tariff Revision (2022 videos)
- DEC_2023_ELEC_01: February 2023 66% Cost-Reflective Electricity Tariff Revision (late Dec 2022 - 2023 videos)
- DEC_2024_ELEC_01: March/July 2024 Downward Electricity Tariff Revisions (2024 videos)

Outputs:
- I:/A_Research Project/Datasets/20_sl_electricity_tariff_decision_comments/sl_tariff_decision_comments.csv
- data/processed/p1_decision_linked_comments.parquet
- data/processed/p1_decision_linked_comments.csv
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.parse
import urllib.request
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "Datasets" / "20_sl_electricity_tariff_decision_comments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SEARCH_QUERIES = [
    # 2022 Aug 75% hike
    "විදුලි බිල වැඩි 2022 අගෝස්තු Ada Derana",
    "විදුලි ගාස්තු සංශෝධනය 2022 Hiru News",
    "විදුලි බිල ඉහළට 2022 Ada Derana",
    "විදුලි බිල 2022 අගෝස්තු Hiru News",
    "විදුලි ගාස්තු ඉහළ 2022 Newsfirst",
    "තුන් හතර ගුණයකින් විදුලි බිල වැඩි විය හැකියි",
    "හෙට සිට වැඩි වන විදුලි බිල 2022 Hiru News",
    "මණ්ඩලය ඉල්ලූ 835%ක විදුලි බිල වැඩි වීම",
    "රුපියල් 360 එක මාසයක විදුලි බිලක් 2022 Newsfirst",
    "විදුලි බිල යළිත් වැඩි වෙයි ද 2022",
    # 2023 Feb 66% hike
    "විදුලි බිල 66% 2023 Ada Derana",
    "විදුලි බිල වැඩි වෙන අලුත් ම විදිය Hiru News 2023",
    "66% කින් විදුලි බිල වැඩි වන හැටි Hiru News",
    "විදුලි ගාස්තු සංශෝධනය අද සිට 2023 Hiru News",
    "විදුලි බිල වැඩිකිරීම සම්බන්ධයෙන් පැවැත්වෙන විශේෂ මාධ්‍ය හමුව 2023",
    "විදුලි බිල වැඩි කරනවාද නැත්නම් පැය හයක් විදුලිය කපනවාද",
    "ජනක රත්නායක විදුලි බිල 2023",
    "කාංචන විජේසේකර විදුලි බිල 2023",
    "විදුලි බිලේ විදුලි සැර 2023 Ada Derana",
    "අද සිට ඒකක අනුව විදුලි ගාස්තු වැඩි වෙන හැටි 2023",
    "විදුලි බිල වැඩි කළේ ලංවිම පොහොසත් කරන්න නෙවෙයි",
    # 2024 Mar/Jul downward revision
    "විදුලි බිල අඩු කරයි 2024 මාර්තු Ada Derana",
    "විදුලි බිල 21.9% අඩු 2024 Hiru News",
    "විදුලි ගාස්තු අඩු කෙරේ 2024 Ada Derana",
    "විදුලි බිල 22.5% අඩු 2024",
    "විදුලි බිල තවත් අඩු කරන්නම් 2024 Ada Derana",
    "විදුලි ගාස්තුවට මොකක් වෙයිද 2024 Ada Derana",
    "විදුලි බිල ගැන ජනතාව තැබූ බලාපොරොත්තු 2024",
    "අඩුවූ විදුලි ගාස්තුව ගැන ජනතා මත 2024 Ada Derana",
    "විදුලි බිල 30%කින් අඩු කරනවා 2024 Hiru News",
]

Tutorial_EXCLUDE_RE = re.compile(
    r"(?:how\s*to|calculation\s*of|ගණනය\s*කිරීම|ගෙදර\s*විදුලි\s*බිල\s*අඩු\s*කරන\s*හැටි|"
    r"සුපිරිම\s*ක්‍රම|tariff\s*chart\s*rate)",
    re.IGNORECASE,
)
CHANNEL_EXCLUDE_RE = re.compile(
    r"(?:electronic|engineering|eco\b|class\s*room|cash\s*logics|mr\s*deda)",
    re.IGNORECASE,
)


def _find_dict_values(obj: Any, target_key: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                yield v
            else:
                yield from _find_dict_values(v, target_key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_dict_values(item, target_key)


def search_youtube_videos(query: str, max_results: int = 8) -> list[dict[str, str]]:
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
    if not m:
        return []
    data = json.loads(m.group(1))
    results = []
    for vr in _find_dict_values(data, "videoRenderer"):
        vid = vr.get("videoId", "")
        title = "".join(r.get("text", "") for r in vr.get("title", {}).get("runs", []))
        channel = "".join(r.get("text", "") for r in vr.get("ownerText", {}).get("runs", []))
        if vid and title:
            results.append({"video_id": vid, "title": title, "channel": channel})
            if len(results) >= max_results:
                break
    return results


def map_publish_date_to_decision(pub_date_iso: str, title: str, channel: str = "") -> str | None:
    """Assign a news broadcast video to DEC_2022_ELEC_01, DEC_2023_ELEC_01, or DEC_2024_ELEC_01."""
    if not pub_date_iso or len(pub_date_iso) < 10:
        return None
    if Tutorial_EXCLUDE_RE.search(title) or CHANNEL_EXCLUDE_RE.search(channel):
        return None
    date_part = pub_date_iso[:10]
    t_low = title.lower()
    if not any(w in t_low for w in ["විදුලි", "electricity", "ceb", "tariff", "power", "බිල", "ගාස්තු", "ඉන්ධන", "සහන"]):
        return None

    # DEC_2022_ELEC_01: Jan 2022 - Dec 15, 2022 (initial 75% tariff revision window)
    if "2022-01-01" <= date_part <= "2022-12-15":
        return "DEC_2022_ELEC_01"
    # DEC_2023_ELEC_01: Dec 16, 2022 - Dec 31, 2023 (Jan 2023 Cabinet memo & Feb 2023 66% revision window)
    if "2022-12-16" <= date_part <= "2023-12-31":
        return "DEC_2023_ELEC_01"
    # DEC_2024_ELEC_01: Jan 1, 2024 - Dec 31, 2024 (2024 downward tariff revision window)
    if "2024-01-01" <= date_part <= "2024-12-31":
        return "DEC_2024_ELEC_01"
    return None


def detect_language_modality(text: str) -> str:
    has_sinhala = bool(re.search(r"[\u0D80-\u0DFF]", text))
    has_latin = bool(re.search(r"[A-Za-z]{3,}", text))
    if has_sinhala and has_latin:
        return "si_en_codemixed"
    if has_sinhala:
        return "si"
    singlish_markers = (
        r"\b(?:api|apita|eka|epa|naha|naa|wadi|adu|mun|munta|rata|bila|gewanna|minissu|"
        r"hora|horu|pissu|mewa|dan|eth|nam|karanna|kanna|bonna|salli|mahajanathawa|"
        r"anduwata|anduwa|kanchana|ranil|ceb|light|kapanawa|kapanna|hena|sapayak)\b"
    )
    if re.search(singlish_markers, text, flags=re.IGNORECASE):
        return "singlish"
    return "en"


def classify_tariff_sentiment_and_stance(text: str, decision_id: str, model_sent: str = "Neutral") -> tuple[str, str]:
    """Domain-calibrated Sinhala/Singlish/English tariff sentiment and policy stance annotator."""
    t = text.lower()

    neg_patterns = (
        r"(?:හෙණ|හෙන|සාප|විනාස|කාලකන්නි|මීහරකා|හොර|හොරු|වංචා|පරයෝ|බොරු|පච|දුක්|"
        r"දරාගන්න\s*බෑ|අමාරු|වැඩි\s*කර|වැඩි\s*වෙන|වැඩිකලොත්|විනාශ|බූරුව|මෝඩ|ගසා\s*කන|හූරන්|"
        r"ණය\s*කන්ද|බලාපොරොත්තු\s*සුන්|එපා|බැහැ|බෑ|මැරෙන්න|වස|පිස්සු|ලැජ්ජ|මකබෑවි|"
        r"මුං|මුන්|තොපි|තොට|යකෝ|බල්ල|බූරු|ගොන්|වඳුර|කඩේ|රැවටු|රවට්ට|අසාධාරණ|බදු|"
        r"ජීවත්\s*වෙන්න|කන්න\s*නෑ|සල්ලි\s*නෑ|බඩගින්න|පාඩු|බෝනස්|වැටුප්|මන්ත්‍රී|ඇමති|"
        r"horu|hora|wancha|boru|pacha|epa|amaru|sapayak|hena|kalakanni|unfair|corrupt|burden|suffer|"
        r"wadi\s*karala|gewanna\s*baa|salli\s*na|munta|thopi|pissu|ballo|gon|moda|"
        r"😡|🤬|😭|😢|👎|🤦|🤮|💩)"
    )
    pos_patterns = (
        r"(?:හොඳයි|හොදයි|සහනයක්|ස්තූති|ගොඩ\s*දාන්න|සතුටු|සාධාරණ|නියමයි|ජය\s*වේවා|"
        r"අගය\s*කරනවා|හොඳ\s*තීරණයක්|අඩු\s*කිරීම\s*හොඳයි|"
        r"sahanayak|hondai|hodai|good|great|thank|welcome\s*move|relief|niyamai|jayawewa)"
    )
    is_sarcastic_thanks = ("ස්තූති" in t or "😂" in t or "😁" in t) and (
        decision_id in ("DEC_2022_ELEC_01", "DEC_2023_ELEC_01")
        or bool(re.search(r"(?:කාන්චනගේ\s*ගෙදරට|මෝල්|බොරු|පච|හූරන්)", t))
    )

    has_neg = bool(re.search(neg_patterns, t)) or is_sarcastic_thanks
    has_pos = bool(re.search(pos_patterns, t)) and not is_sarcastic_thanks and not has_neg

    if has_neg:
        sentiment = "Negative"
    elif has_pos:
        sentiment = "Positive"
    elif model_sent in ("Negative", "Positive"):
        sentiment = model_sent
    else:
        sentiment = "Neutral"

    if decision_id in ("DEC_2022_ELEC_01", "DEC_2023_ELEC_01"):
        if sentiment == "Negative" or re.search(r"(?:අඩු\s*කරන්න|adu\s*karanna|කපන්න|එපා)", t):
            stance = "Oppose"
        elif sentiment == "Positive":
            stance = "Support"
        else:
            stance = "Neutral_or_Unclear"
    else:  # DEC_2024_ELEC_01 (Downward revision / relief debate)
        if sentiment == "Positive" or re.search(r"(?:ගොඩ\s*දාන්න|පච\s*කිව්වේ\s*නැ|සහන)", t):
            stance = "Support"
        elif sentiment == "Negative":
            stance = "Oppose"
        else:
            stance = "Neutral_or_Unclear"

    return sentiment, stance


def fetch_video_comments(video_id: str, max_pages: int = 5) -> tuple[str, list[dict[str, str]]]:
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

    comments: list[dict[str, str]] = []
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
            cid = props.get("commentId", "")
            content = props.get("content", {}).get("content", "").strip()
            pub_time = props.get("publishedTime", "")
            if cid and content and len(content) >= 4:
                comments.append({"comment_id": f"YT_{cid}", "text": content, "relative_time": pub_time})

        for cr in _find_dict_values(cresp, "commentRenderer"):
            cid = cr.get("commentId", "")
            content = "".join(r.get("text", "") for r in cr.get("contentText", {}).get("runs", [])).strip()
            pub_time = "".join(r.get("text", "") for r in cr.get("publishedTimeText", {}).get("runs", []))
            if cid and content and len(content) >= 4:
                comments.append({"comment_id": f"YT_{cid}", "text": content, "relative_time": pub_time})

        for cmd in _find_dict_values(cresp, "continuationCommand"):
            ntok = cmd.get("token", "")
            if ntok and ntok not in visited and len(ntok) > 40:
                queue.append(ntok)

    dedup = {c["comment_id"]: c for c in comments}
    return pub_date, list(dedup.values())


def build_decision_comment_dataset() -> pd.DataFrame:
    from src.pipeline1_public_opinion.predict import get_sentiment_model
    sent_clf = get_sentiment_model()

    seen_videos: dict[str, dict[str, str]] = {}
    for q in SEARCH_QUERIES:
        for v in search_youtube_videos(q, max_results=7):
            seen_videos[v["video_id"]] = v

    print(f"Discovered {len(seen_videos)} candidate news videos. Inspecting publish dates & extracting comments...")

    records: list[dict[str, Any]] = []
    for vid, vmeta in seen_videos.items():
        pub_date, comms = fetch_video_comments(vid, max_pages=6)
        dec_id = map_publish_date_to_decision(pub_date, vmeta["title"], vmeta["channel"])
        if not dec_id or not comms:
            continue
        print(f"  [{dec_id}] {pub_date[:10]} | {vid} | {vmeta['channel']} | {vmeta['title'][:55]} -> {len(comms)} comments")
        texts = [re.sub(r"\s+", " ", c["text"]).strip() for c in comms]
        model_preds = sent_clf.predict(texts)
        for c, text_clean, m_sent in zip(comms, texts, model_preds):
            if len(text_clean) < 4:
                continue
            lang = detect_language_modality(text_clean)
            sent, stance = classify_tariff_sentiment_and_stance(text_clean, dec_id, str(m_sent))
            records.append({
                "comment_id": c["comment_id"],
                "decision_id": dec_id,
                "video_id": vid,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "video_publish_date": pub_date[:10],
                "channel": vmeta["channel"],
                "video_title": vmeta["title"],
                "text": text_clean,
                "language": lang,
                "sentiment_label": sent,
                "stance_label": stance,
                "source_dataset": "sl_electricity_tariff_youtube_news_2022_2024",
            })

    df = pd.DataFrame(records).drop_duplicates(subset=["comment_id"]).reset_index(drop=True)

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv_path = DATASETS_ROOT / "sl_tariff_decision_comments.csv"
    proc_csv_path = PROCESSED_DIR / "p1_decision_linked_comments.csv"
    proc_pq_path = PROCESSED_DIR / "p1_decision_linked_comments.parquet"

    df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    df.to_csv(proc_csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(proc_pq_path, index=False)

    print(f"\nSaved {len(df)} decision-linked comments across {df['video_id'].nunique()} dated news broadcasts:")
    print(df.groupby(["decision_id", "stance_label"]).size().unstack(fill_value=0))
    print("\nSentiment breakdown by decision:")
    print(df.groupby(["decision_id", "sentiment_label"]).size().unstack(fill_value=0))
    return df


def import_saved_facebook_exports(fb_dir: Path | None = None) -> pd.DataFrame:
    """Parse user-saved Facebook HTML/JSON/CSV files from `facebook_saved_posts/` and merge with dataset."""
    from src.pipeline1_public_opinion.predict import get_sentiment_model
    fb_folder = fb_dir or (DATASETS_ROOT / "facebook_saved_posts")
    fb_folder.mkdir(parents=True, exist_ok=True)

    proc_pq_path = PROCESSED_DIR / "p1_decision_linked_comments.parquet"
    base_df = pd.read_parquet(proc_pq_path) if proc_pq_path.exists() else pd.DataFrame()

    raw_items: list[dict[str, str]] = []
    # 1. JSON files exported via DevTools or scraped live
    for jf in fb_folder.glob("*.json"):
        data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        post_list = data.get("posts", [data]) if isinstance(data, dict) else [{"comments": data}]
        for p_idx, post_obj in enumerate(post_list):
            p_url = post_obj.get("post_url", "https://www.facebook.com/")
            p_date = post_obj.get("date", "2023-02-15")[:10]
            p_chan = post_obj.get("channel", "Facebook News Page")
            p_title = post_obj.get("title", jf.stem)
            p_dec = post_obj.get("decision_id", data.get("decision_id", "DEC_2023_ELEC_01"))
            items = post_obj.get("comments", [])
            for c_idx, it in enumerate(items):
                txt = (it.get("text") if isinstance(it, dict) else str(it)).strip()
                if len(txt) >= 4:
                    raw_items.append({
                        "comment_id": f"FB_{jf.stem}_{p_idx}_{c_idx:04d}",
                        "decision_id": p_dec,
                        "video_id": f"FB_{jf.stem}_{p_idx}",
                        "video_url": p_url,
                        "video_publish_date": p_date,
                        "channel": p_chan,
                        "video_title": p_title,
                        "text": re.sub(r"\s+", " ", txt),
                    })

    # 2. Saved HTML pages (Ctrl+S in Edge/Chrome)
    for hf in fb_folder.glob("*.htm*"):
        html_text = hf.read_text(encoding="utf-8", errors="ignore")
        # Extract text blocks inside dir="auto" divs (standard Facebook comment text container)
        blocks = re.findall(r'<div[^>]+dir="auto"[^>]*>(.*?)</div>', html_text, flags=re.DOTALL)
        for idx, blk in enumerate(blocks):
            clean = re.sub(r"<[^>]+>", " ", blk)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) >= 8 and not clean.startswith(("http", "Write a comment", "Most relevant")):
                raw_items.append({
                    "comment_id": f"FB_{hf.stem}_{idx:04d}",
                    "decision_id": "DEC_2023_ELEC_01",
                    "video_id": f"FB_{hf.stem}",
                    "video_url": "https://www.facebook.com/",
                    "video_publish_date": "2023-02-15",
                    "channel": "Facebook Saved Post",
                    "video_title": hf.stem,
                    "text": clean,
                })

    if not raw_items:
        print(f"No saved Facebook files found in: {fb_folder}")
        return base_df

    sent_clf = get_sentiment_model()
    preds = sent_clf.predict([r["text"] for r in raw_items])
    fb_records = []
    for r, m_sent in zip(raw_items, preds):
        lang = detect_language_modality(r["text"])
        sent, stance = classify_tariff_sentiment_and_stance(r["text"], r["decision_id"], str(m_sent))
        fb_records.append({
            **r,
            "language": lang,
            "sentiment_label": sent,
            "stance_label": stance,
            "source_dataset": "sl_electricity_tariff_facebook_saved_posts",
        })

    fb_df = pd.DataFrame(fb_records)
    merged = pd.concat([base_df, fb_df], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    merged.to_parquet(proc_pq_path, index=False)
    merged.to_csv(PROCESSED_DIR / "p1_decision_linked_comments.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(DATASETS_ROOT / "sl_tariff_decision_comments.csv", index=False, encoding="utf-8-sig")
    print(f"Merged {len(fb_df)} Facebook comments! Total dataset size: {len(merged)}")
    return merged


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--import-fb" in sys.argv:
        import_saved_facebook_exports()
    else:
        build_decision_comment_dataset()
