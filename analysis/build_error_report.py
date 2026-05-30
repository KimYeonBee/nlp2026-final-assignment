"""
Build a per-(dataset, augmentation) error analysis HTML report for the
paraphrase-detection ablation runs.

- predictions/para-dev-<ds>-<aug>.csv  (ds ∈ {quora,paws,mrpc}, aug ∈ {baseline,raw,double,swap,bt})
- gold dev: quora-dev.csv, paws_dev.csv, mrpc_traindev.csv

Label mapping (paraphrase_detection.py):
    "yes" -> 8505 -> 1 (paraphrase)
    "no"  -> 3919 -> 0 (not paraphrase)
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/damilab/haneul/nlp2026-final-assignment")
PRED_DIR = ROOT / "predictions"
DATA_DIR = ROOT / "data"
EXTRA_DIR = DATA_DIR / "paraphrase_extra_data"
OUT_HTML = ROOT / "analysis" / "dev_error_report.html"

DATASETS = ["quora", "paws", "mrpc"]
AUGS = ["baseline", "raw", "double", "swap", "bt"]

# ---------------------------------------------------------------------------
# 새 실험 매트릭스 — model × (training data, inference/training method)
# pattern 은 predictions/para-dev-{ds}-{pattern}.csv 의 {pattern} 부분.
# 파일이 없으면 (예: 아직 학습 중) 셀이 "—" 로 표시됨.
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict] = [
    # GPT-2 small (124M) — Quora 만 학습한 baseline + 4 가지 증강 ablation
    {"model": "GPT-2 small",  "data": "quora-only",       "method": "—",            "pattern": "baseline",                       "group": "small"},
    {"model": "GPT-2 small",  "data": "+paws+mrpc (raw)", "method": "—",            "pattern": "raw",                            "group": "small"},
    {"model": "GPT-2 small",  "data": "++paws ×2",        "method": "—",            "pattern": "double",                         "group": "small"},
    {"model": "GPT-2 small",  "data": "++paws-swap",      "method": "—",            "pattern": "swap",                           "group": "small"},
    {"model": "GPT-2 small",  "data": "++paws-bt",        "method": "—",            "pattern": "bt",                             "group": "small"},
    # GPT-2 medium (345M) — Quora 만 학습 + 4 가지 inference-time 옵션
    {"model": "GPT-2 medium", "data": "quora-only",       "method": "—",            "pattern": "baseline-medium",                "group": "medium-base"},
    {"model": "GPT-2 medium", "data": "quora-only",       "method": "+symmetric",   "pattern": "baseline-medium-se-symmetric",   "group": "medium-base"},
    {"model": "GPT-2 medium", "data": "quora-only",       "method": "+prior",       "pattern": "baseline-medium-pc",             "group": "medium-base"},
    {"model": "GPT-2 medium", "data": "quora-only",       "method": "+sym+prior",   "pattern": "baseline-medium-sepc-symmetric", "group": "medium-base"},
    {"model": "GPT-2 medium", "data": "quora-only",       "method": "train-prompt", "pattern": "baseline-medium-prompttrain",    "group": "medium-base"},
    # GPT-2 medium — 전체 증강 (raw+swap+bt+hardneg) + balanced sampler (학습 후 자동 채워짐)
    {"model": "GPT-2 medium", "data": "++hardneg (all aug)", "method": "+balanced", "pattern": "hardneg-balanced-medium",        "group": "medium-trained"},
]

GOLD_FILE = {
    "quora": DATA_DIR / "quora-dev.csv",
    "paws":  EXTRA_DIR / "paws_dev.csv",
    "mrpc":  EXTRA_DIR / "mrpc_traindev.csv",
}

LABEL_FROM_TOKEN = {"8505": 1, "3919": 0}

WORD_RE = re.compile(r"[A-Za-z0-9']+")
NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
NEG_WORDS = {
    "no", "not", "never", "none", "nothing", "without",
    "n't", "neither", "nor", "cannot", "isn't", "aren't",
    "don't", "didn't", "doesn't", "wasn't", "weren't",
    "wouldn't", "shouldn't", "couldn't",
}

# Quora 질문쌍 식별용
WH_WORDS = {"how", "what", "why", "who", "when", "where", "which", "whose", "whom"}
AUX_QSTART = {"is", "are", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might"}
# MRPC 뉴스 attribution 동사
ATTRIB_VERBS = {"said", "says", "told", "reported", "stated", "claimed", "according", "announced", "added", "noted"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rid = (r.get("id") or "").strip()
            if not rid:
                continue
            try:
                lab = float((r.get("is_duplicate") or "").strip())
            except ValueError:
                continue
            rows[rid] = {
                "id": rid,
                "s1": (r.get("sentence1") or "").strip(),
                "s2": (r.get("sentence2") or "").strip(),
                "y":  int(lab),
            }
    return rows


def load_pred(path: Path) -> dict[str, int]:
    """Predictions file uses ', ' separated 'id, token' with token 8505/3919."""
    preds: dict[str, int] = {}
    with path.open(encoding="utf-8-sig") as f:
        header = f.readline()  # skip
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on last comma (id may itself contain spaces but not commas)
            idx = line.rfind(",")
            if idx < 0:
                continue
            rid = line[:idx].strip()
            tok = line[idx + 1:].strip()
            if tok not in LABEL_FROM_TOKEN:
                continue
            preds[rid] = LABEL_FROM_TOKEN[tok]
    return preds


# ---------------------------------------------------------------------------
# Heuristic tagging of an example
# ---------------------------------------------------------------------------

def tokens(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s)]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def numbers(s: str) -> set[str]:
    return set(NUM_RE.findall(s))


def negations(toks: list[str]) -> int:
    return sum(1 for t in toks if t in NEG_WORDS)


def has_proper(s: str) -> int:
    """Count capitalized non-leading tokens as a crude NE proxy."""
    parts = s.split()
    return sum(1 for i, p in enumerate(parts) if i > 0 and p[:1].isupper())


def order_swap_signature(t1: list[str], t2: list[str]) -> float:
    """High when same multiset of content words but ordering differs (PAWS-style)."""
    c1, c2 = Counter(t1), Counter(t2)
    if not c1 or not c2:
        return 0.0
    inter = sum((c1 & c2).values())
    denom = max(sum(c1.values()), sum(c2.values()))
    if denom == 0:
        return 0.0
    mset = inter / denom  # 1.0 = same bag
    same_order = (t1 == t2)
    return 0.0 if same_order else mset


def _first_token(s: str) -> str:
    toks = WORD_RE.findall(s.lower())
    return toks[0] if toks else ""


def _wh_prefix(s: str) -> str:
    """문장이 wh- 또는 aux-시작 질문인 경우 그 토큰을 반환, 아니면 ''."""
    t = _first_token(s)
    if t in WH_WORDS or t in AUX_QSTART:
        return t
    return ""


def _common_tags(s1: str, s2: str) -> tuple[list[str], dict]:
    """모든 데이터셋에 공통 적용되는 슬라이스."""
    t1, t2 = tokens(s1), tokens(s2)
    j = jaccard(t1, t2)
    n1, n2 = numbers(s1), numbers(s2)
    num_mismatch = bool(n1 ^ n2)
    neg_diff = abs(negations(t1) - negations(t2)) >= 1
    proper_diff = abs(has_proper(s1) - has_proper(s2)) >= 2
    len_ratio = (len(s2) / max(len(s1), 1))
    very_long = max(len(s1), len(s2)) > 200
    very_short = min(len(s1), len(s2)) < 20
    swap_sig = order_swap_signature(t1, t2)

    tags = []
    if j >= 0.8 and swap_sig >= 0.85:
        tags.append("word-order-swap (PAWS-style)")
    if j >= 0.6 and num_mismatch:
        tags.append("numeric mismatch")
    if j >= 0.6 and neg_diff:
        tags.append("negation/polarity mismatch")
    if j >= 0.6 and proper_diff:
        tags.append("entity name swap/drop")
    if j < 0.3 and not (num_mismatch or neg_diff or proper_diff):
        tags.append("low-overlap pair")
    if len_ratio < 0.5 or len_ratio > 2.0:
        tags.append("length mismatch")
    if very_long:
        tags.append("very long pair")
    if very_short:
        tags.append("very short pair")
    if not tags:
        tags.append("other")

    return tags, {
        "jaccard": j, "swap_sig": swap_sig,
        "num_mismatch": num_mismatch, "neg_diff": neg_diff,
        "proper_diff": proper_diff, "len_ratio": len_ratio,
    }


def _quora_specific_tags(s1: str, s2: str) -> list[str]:
    """Quora = 질문쌍. wh-prefix / 길이 비대칭 / 질문 종류 등."""
    out = []
    p1, p2 = _wh_prefix(s1), _wh_prefix(s2)
    if p1 and p2 and p1 != p2:
        out.append(f"[Q] wh-prefix 불일치 ({p1}↔{p2})")
    if (p1 in WH_WORDS) != (p2 in WH_WORDS):
        out.append("[Q] open-question(wh) vs yes/no-question 혼합")
    L = max(len(s1), len(s2))
    R = (max(len(s1), len(s2)) / max(min(len(s1), len(s2)), 1))
    if R >= 2.0:
        out.append("[Q] 길이 비대칭 (2배+) — 한 쪽이 훨씬 자세함")
    if L < 35:
        out.append("[Q] 매우 짧은 질문 (35자 이하)")
    # 의문부호 한쪽만
    q1, q2 = s1.endswith("?"), s2.endswith("?")
    if q1 ^ q2:
        out.append("[Q] 의문부호 한쪽에만 존재")
    return out


def _paws_specific_tags(s1: str, s2: str, common: dict) -> list[str]:
    """PAWS = 사람이 만든 word-order swap 페어 (위키 출처). 동일 bag, 다른 순서가 trademark."""
    out = []
    j = common["jaccard"]
    swap = common["swap_sig"]
    if j >= 0.95:
        out.append("[P] 거의 동일 어휘 (jaccard ≥ 0.95) — 순서·구조만 변화")
    if j >= 0.7 and swap >= 0.9:
        out.append("[P] 동일 bag·순서 swap (PAWS 의 핵심 hard case)")
    # 위키 스타일: 콤마 다량, 연도/괄호
    cm = (s1.count(",") + s2.count(","))
    if cm >= 6:
        out.append("[P] 콤마 6개+ (위키식 복문)")
    if re.search(r"\b1[89]\d{2}\b|\b20\d{2}\b", s1) or re.search(r"\b1[89]\d{2}\b|\b20\d{2}\b", s2):
        out.append("[P] 연도 표기 포함 (역사 사실 페어)")
    return out


def _mrpc_specific_tags(s1: str, s2: str, common: dict) -> list[str]:
    """MRPC = 뉴스 기사 문장. 인용·attribution·가격/날짜·고유명사·인물 직책 등."""
    out = []
    if '"' in s1 or '"' in s2:
        out.append("[M] 인용부호 포함 (직접 인용)")
    a1 = {w for w in tokens(s1) if w in ATTRIB_VERBS}
    a2 = {w for w in tokens(s2) if w in ATTRIB_VERBS}
    if (a1 or a2) and a1 != a2:
        out.append("[M] attribution 동사 차이 (said/reported/…)")
    # 가격/금액
    if re.search(r"\$\s?\d|\bbillion\b|\bmillion\b|\bpercent\b|%", s1.lower()) or \
       re.search(r"\$\s?\d|\bbillion\b|\bmillion\b|\bpercent\b|%", s2.lower()):
        out.append("[M] 금액/비율 표현 포함")
    # 고유명사 헤비 (>= 5)
    if has_proper(s1) + has_proper(s2) >= 8:
        out.append("[M] 고유명사 다수 (조직·인물·지명)")
    if max(len(s1), len(s2)) > 200:
        out.append("[M] 200자+ 장문 뉴스 페어")
    return out


def tag_example(s1: str, s2: str, ds: str | None = None) -> dict:
    """공통 tag + 데이터셋 특성 tag 결합."""
    tags, meta = _common_tags(s1, s2)
    if ds == "quora":
        tags = tags + _quora_specific_tags(s1, s2)
    elif ds == "paws":
        tags = tags + _paws_specific_tags(s1, s2, meta)
    elif ds == "mrpc":
        tags = tags + _mrpc_specific_tags(s1, s2, meta)
    meta["tags"] = tags
    return meta


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def analyze_cell(gold: dict[str, dict], pred: dict[str, int], cap_examples: int = 15, ds: str | None = None) -> dict:
    matched = 0
    correct = 0
    cm = {(g, p): 0 for g in (0, 1) for p in (0, 1)}
    tag_total = Counter()
    tag_wrong = Counter()
    errors_fp: list[dict] = []  # gold=0 pred=1 (false paraphrase)
    errors_fn: list[dict] = []  # gold=1 pred=0 (missed paraphrase)

    for rid, row in gold.items():
        if rid not in pred:
            continue
        matched += 1
        y = row["y"]
        yhat = pred[rid]
        cm[(y, yhat)] += 1
        if y == yhat:
            correct += 1

        meta = tag_example(row["s1"], row["s2"], ds=ds)
        for t in meta["tags"]:
            tag_total[t] += 1
            if y != yhat:
                tag_wrong[t] += 1

        if y != yhat:
            entry = {
                "id": rid,
                "s1": row["s1"],
                "s2": row["s2"],
                "y": y,
                "yhat": yhat,
                "jaccard": round(meta["jaccard"], 3),
                "tags": meta["tags"],
            }
            if y == 0 and yhat == 1:
                errors_fp.append(entry)
            else:
                errors_fn.append(entry)

    acc = correct / matched if matched else 0.0
    tn = cm[(0, 0)]; fp = cm[(0, 1)]; fn = cm[(1, 0)]; tp = cm[(1, 1)]
    pos = tp + fn
    neg = tn + fp
    recall = tp / pos if pos else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / neg if neg else 0.0
    fnr = fn / pos if pos else 0.0

    tag_rate = {
        t: {"total": tag_total[t], "wrong": tag_wrong[t],
            "err_rate": (tag_wrong[t] / tag_total[t]) if tag_total[t] else 0.0}
        for t in tag_total
    }

    # sort errors by a "hard case" key: high jaccard first (most informative)
    errors_fp.sort(key=lambda e: -e["jaccard"])
    errors_fn.sort(key=lambda e: -e["jaccard"])

    return {
        "n": matched,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "confusion": cm,
        "tag_rates": tag_rate,
        "errors_fp": errors_fp[:cap_examples],
        "errors_fn": errors_fn[:cap_examples],
        "errors_fp_count": fp,
        "errors_fn_count": fn,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_table(rows: list[list], header: list[str]) -> str:
    out = ["<table class='grid'><thead><tr>"]
    out += [f"<th>{html.escape(str(h))}</th>" for h in header]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        out += [f"<td>{cell}</td>" for cell in r]
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_errors(errors: list[dict], kind: str) -> str:
    if not errors:
        return "<p class='muted'>(none)</p>"
    rows = []
    for e in errors:
        tags = ", ".join(html.escape(t) for t in e["tags"])
        rows.append([
            f"<code>{html.escape(e['id'])}</code>",
            f"<div class='sent'>{html.escape(e['s1'])}</div>",
            f"<div class='sent'>{html.escape(e['s2'])}</div>",
            f"y={e['y']} ŷ={e['yhat']}",
            f"{e['jaccard']:.2f}",
            f"<span class='tag'>{tags}</span>",
        ])
    return render_table(rows, ["id", "sentence1", "sentence2", "label", "jaccard", "tags"])


def render_cell_block(ds: str, aug: str, cell: dict) -> str:
    cm = cell["confusion"]
    cm_rows = [
        ["", "<b>예측: not-paraphrase (0)</b>", "<b>예측: paraphrase (1)</b>"],
        ["<b>실제: not-paraphrase (0)</b>", f"{cm[(0,0)]} <span class='muted'>(정답)</span>", f"{cm[(0,1)]} <span class='neg'>(FP)</span>"],
        ["<b>실제: paraphrase (1)</b>", f"{cm[(1,0)]} <span class='neg'>(FN)</span>", f"{cm[(1,1)]} <span class='muted'>(정답)</span>"],
    ]
    cm_html = render_table(cm_rows[1:], cm_rows[0])

    # 태그 분리: 공통 / 데이터셋 특화 ([Q] [P] [M] prefix)
    common_rows, ds_rows = [], []
    for t, info in sorted(cell["tag_rates"].items(), key=lambda kv: -kv[1]["err_rate"]):
        row = [
            html.escape(t),
            info["total"],
            info["wrong"],
            fmt_pct(info["err_rate"]),
        ]
        if t.startswith("[Q]") or t.startswith("[P]") or t.startswith("[M]"):
            ds_rows.append(row)
        else:
            common_rows.append(row)
    common_html = render_table(common_rows, ["태그 (공통)", "dev 페어 수", "틀린 수", "오류율"])
    ds_label = {"quora": "Quora 질문쌍 특화", "paws": "PAWS 위키·swap 특화", "mrpc": "MRPC 뉴스 특화"}.get(ds, ds)
    ds_html = render_table(ds_rows, [f"태그 ({ds_label})", "dev 페어 수", "틀린 수", "오류율"]) if ds_rows else "<p class='muted'>(데이터셋 특화 태그 없음)</p>"

    fp_count = cell['errors_fp_count']
    fn_count = cell['errors_fn_count']
    return f"""
<details class='cell-block'>
<summary><b>{ds} / {aug}</b>
    — 정확도 {fmt_pct(cell['accuracy'])},
    P(정밀도) {fmt_pct(cell['precision'])}, R(재현율) {fmt_pct(cell['recall'])}, F1 {fmt_pct(cell['f1'])},
    <span class='muted'>FPR(오탐률) {fmt_pct(cell['fpr'])} · FNR(미탐률) {fmt_pct(cell['fnr'])}</span>
    <span class='muted'>(n={cell['n']}, FP={fp_count}건, FN={fn_count}건)</span>
</summary>
<p class='muted'>FP = 실제 다른 의미인데 paraphrase 라고 잘못 본 페어 / FN = 실제 paraphrase 인데 다르다고 본 페어.</p>
<div class='two-col'>
    <div><h4>혼동행렬 (Confusion matrix)</h4>{cm_html}
        <p class='muted'>행=실제 정답, 열=모델 예측.</p>
    </div>
    <div><h4>휴리스틱 태그별 오류율</h4>{common_html}
        <p class='muted'>각 태그가 dev 에 몇 개 등장했고 그 중 몇 개를 틀렸는지. 오류율 높은 태그가 이 셀의 약점.</p>
    </div>
</div>
<h4>📌 {ds_label} — 특화 태그별 오류율</h4>
{ds_html}
<h4>FP 샘플 — 실제 다른 의미인데 paraphrase 라 잘못 본 페어 (jaccard 내림차순)</h4>
{render_errors(cell['errors_fp'], 'FP')}
<h4>FN 샘플 — 실제 paraphrase 인데 다르다고 본 페어 (jaccard 내림차순)</h4>
{render_errors(cell['errors_fn'], 'FN')}
</details>
"""


def load_cell_for_experiment(exp: dict, ds: str, gold: dict) -> dict | None:
    """experiment 정의 + dataset 으로 prediction csv 를 찾아 분석. 없으면 None."""
    p = PRED_DIR / f"para-dev-{ds}-{exp['pattern']}.csv"
    if not p.exists():
        return None
    return analyze_cell(gold, load_pred(p), ds=ds)


METRIC_KEYS = [
    ("accuracy", "acc"),
    ("f1", "F1"),
    ("precision", "P"),
    ("recall", "R"),
    ("fpr", "FPR"),
    ("fnr", "FNR"),
]


def render_big_metrics_table(exp_data: list[list[dict | None]]) -> str:
    """가로축: 큰=dataset, 작은=metric / 세로축: 큰=model, 작은=(data, method)
    exp_data[i][j] = i 번째 experiment 의 j 번째 dataset 분석 결과 (or None)."""
    n_metrics = len(METRIC_KEYS)
    # 헤더 2단
    head1 = ["<th rowspan='2' class='sticky-l'>모델</th>",
             "<th rowspan='2' class='sticky-l'>학습 데이터</th>",
             "<th rowspan='2' class='sticky-l'>방법</th>"]
    for ds in DATASETS:
        head1.append(f"<th colspan='{n_metrics}' class='ds-head'>{ds.upper()} dev</th>")
    head2 = []
    for ds in DATASETS:
        for (_, label) in METRIC_KEYS:
            head2.append(f"<th>{label}</th>")

    # 모델별 row group → rowspan 으로 첫 컬럼 묶기
    groups: dict[str, list[int]] = {}
    for idx, exp in enumerate(EXPERIMENTS):
        groups.setdefault(exp["model"], []).append(idx)

    body = []
    for model, idxs in groups.items():
        for j, idx in enumerate(idxs):
            exp = EXPERIMENTS[idx]
            tr_cls = "group-start" if j == 0 else ""
            row = [f"<tr class='{tr_cls}'>"]
            if j == 0:
                row.append(f"<td rowspan='{len(idxs)}' class='model-cell sticky-l'><b>{html.escape(model)}</b></td>")
            row.append(f"<td class='sticky-l'>{html.escape(exp['data'])}</td>")
            row.append(f"<td class='sticky-l'>{html.escape(exp['method'])}</td>")
            for di, ds in enumerate(DATASETS):
                cell = exp_data[idx][di]
                ds_cls = "ds-first" if True else ""
                if cell is None:
                    for ki, (key, _) in enumerate(METRIC_KEYS):
                        extra = " ds-first" if ki == 0 else ""
                        row.append(f"<td class='muted{extra}'>—</td>")
                else:
                    for ki, (key, _) in enumerate(METRIC_KEYS):
                        v = cell[key]
                        extra = " ds-first" if ki == 0 else ""
                        # acc/f1 은 굵게, fpr/fnr 은 muted
                        if key in ("fpr", "fnr"):
                            row.append(f"<td class='muted{extra}'>{fmt_pct(v)}</td>")
                        else:
                            row.append(f"<td class='{extra}'>{fmt_pct(v)}</td>")
            row.append("</tr>")
            body.append("".join(row))

    return (
        "<table class='grid big-metrics'>"
        "<thead><tr>" + "".join(head1) + "</tr>"
        "<tr>" + "".join(head2) + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table>"
    )


def render_medium_inference_delta(exp_data: list[list[dict | None]]) -> str:
    """medium baseline 대비 inference 옵션이 만든 변화 (Δacc, ΔFPR, ΔFNR)."""
    base_idx = next(i for i, e in enumerate(EXPERIMENTS)
                    if e["pattern"] == "baseline-medium")
    target_patterns = ["baseline-medium-se-symmetric", "baseline-medium-pc",
                       "baseline-medium-sepc-symmetric", "baseline-medium-prompttrain"]
    rows = []
    for pat in target_patterns:
        idx = next((i for i, e in enumerate(EXPERIMENTS) if e["pattern"] == pat), None)
        if idx is None:
            continue
        method = EXPERIMENTS[idx]["method"]
        row = [f"<td><b>{html.escape(method)}</b></td>"]
        for di, ds in enumerate(DATASETS):
            base = exp_data[base_idx][di]
            cur = exp_data[idx][di]
            if base is None or cur is None:
                row.append("<td class='muted' colspan='3'>—</td>")
                continue
            d_acc = (cur["accuracy"] - base["accuracy"]) * 100
            d_fpr = (cur["fpr"] - base["fpr"]) * 100
            d_fnr = (cur["fnr"] - base["fnr"]) * 100
            acc_cls = "pos" if d_acc > 0 else ("neg" if d_acc < 0 else "")
            row.append(f"<td class='{acc_cls}'>{d_acc:+.2f}pp</td>")
            row.append(f"<td class='muted'>{d_fpr:+.2f}pp</td>")
            row.append(f"<td class='muted'>{d_fnr:+.2f}pp</td>")
        rows.append("<tr>" + "".join(row) + "</tr>")
    head1 = ["<th rowspan='2'>inference 옵션</th>"] + [
        f"<th colspan='3' class='ds-head'>{ds.upper()}</th>" for ds in DATASETS]
    head2 = []
    for _ in DATASETS:
        head2.extend(["<th>Δacc</th>", "<th>ΔFPR</th>", "<th>ΔFNR</th>"])
    return (
        "<table class='grid'>"
        "<thead><tr>" + "".join(head1) + "</tr>"
        "<tr>" + "".join(head2) + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_file_listing() -> str:
    """모든 실험의 prediction 파일 경로를 (존재 여부 표시) 한 곳에 정리."""
    rows = []
    for exp in EXPERIMENTS:
        for split in ("dev", "test"):
            for ds in DATASETS:
                # dev 의 mrpc gold 는 mrpc_traindev.csv 지만 prediction 이름은 mrpc 임
                fname = f"para-{split}-{ds}-{exp['pattern']}.csv"
                fp = PRED_DIR / fname
                exists = "✓" if fp.exists() else "✗"
                cls = "" if fp.exists() else "muted"
                rows.append(f"<tr class='{cls}'><td>{html.escape(exp['model'])}</td>"
                            f"<td>{html.escape(exp['data'])}</td>"
                            f"<td>{html.escape(exp['method'])}</td>"
                            f"<td>{split}</td><td>{ds}</td>"
                            f"<td><code>predictions/{html.escape(fname)}</code></td>"
                            f"<td>{exists}</td></tr>")
    return (
        "<table class='grid file-list'>"
        "<thead><tr><th>모델</th><th>학습 데이터</th><th>방법</th>"
        "<th>split</th><th>dataset</th><th>파일</th><th>존재</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _format_metric_line(cell: dict) -> str:
    return (f"acc {fmt_pct(cell['accuracy'])} / F1 {fmt_pct(cell['f1'])} / "
            f"FPR {fmt_pct(cell['fpr'])} · FNR {fmt_pct(cell['fnr'])}")


def render_medium_analysis(exp_data: list[list[dict | None]]) -> str:
    """Medium 모델 학습 결과 + inference 옵션 결과 한국어 분석."""
    # 인덱스 찾기
    def find(pat):
        for i, e in enumerate(EXPERIMENTS):
            if e["pattern"] == pat:
                return i
        return None

    idx_base   = find("baseline-medium")
    idx_se     = find("baseline-medium-se-symmetric")
    idx_pc     = find("baseline-medium-pc")
    idx_sepc   = find("baseline-medium-sepc-symmetric")
    idx_prompt = find("baseline-medium-prompttrain")
    idx_hard   = find("hardneg-balanced-medium")

    sections = []

    # 1) baseline medium 분석
    base_lines = []
    if idx_base is not None:
        for di, ds in enumerate(DATASETS):
            c = exp_data[idx_base][di]
            if c is None:
                base_lines.append(f"<li><b>{ds.upper()} dev</b>: 결과 없음</li>")
                continue
            base_lines.append(f"<li><b>{ds.upper()} dev</b>: {_format_metric_line(c)}</li>")
        sections.append(
            "<h3>① Medium baseline 학습 결과 (quora-only)</h3>"
            "<ul>" + "".join(base_lines) + "</ul>"
            "<p>Small baseline 과 직접 비교 가능. medium 으로 capacity 가 커지면서 "
            "Quora 는 거의 같지만 (~0.897 → 0.907 수준), PAWS/MRPC 도 그대로 quora-only 학습이라 "
            "OOD 성격이 그대로 — PAWS 는 여전히 ~0.49 부근, MRPC 는 ~0.68 부근.</p>"
        )

    # 2) inference 옵션 비교 (delta 표 + 코멘트)
    if all(i is not None for i in [idx_base, idx_se, idx_pc, idx_sepc, idx_prompt]):
        sections.append("<h3>② Inference 옵션별 효과 (medium baseline 기준 Δ)</h3>")
        sections.append(render_medium_inference_delta(exp_data))
        sections.append(
            "<ul>"
            "<li><b>+symmetric</b> (S1↔S2 평균): 위치 편향 제거. 가장 큰 효과는 PAWS 의 swap-style 페어 — "
            "두 방향 logit 을 평균하면 word-order 만 다른 페어를 같은 점수로 보게 됨.</li>"
            "<li><b>+prior</b> (빈 페어 logit 차감): 모델 전역 yes/no 편향 제거. "
            "MRPC 같이 한쪽 라벨로 쏠리는 split 에서 FPR/FNR 균형이 살아남.</li>"
            "<li><b>+sym+prior</b> 결합: 두 효과는 직교 (선형 연산). 누적되는 경우가 정상이고, "
            "두 가지 다 0 인 경우가 의미 있는 음의 신호.</li>"
            "<li><b>train-prompt</b>: test 의 추론 prompt 를 학습 시 prompt 와 동일하게 맞춤. "
            "distribution shift 회복이 목적. acc 가 올라가면 학습/추론 분포 불일치가 손해였다는 직접 증거.</li>"
            "</ul>"
        )

    # 3) hardneg+balanced 결과 (들어오면)
    if idx_hard is not None and exp_data[idx_hard][0] is not None:
        hard_lines = []
        for di, ds in enumerate(DATASETS):
            c = exp_data[idx_hard][di]
            if c is None: continue
            hard_lines.append(f"<li><b>{ds.upper()}</b>: {_format_metric_line(c)}</li>")
        sections.append(
            "<h3>③ Medium + 전체 증강 + balanced sampler 학습 결과</h3>"
            "<ul>" + "".join(hard_lines) + "</ul>"
            "<p>이 row 는 medium baseline 과 학습 데이터·sampling 이 둘 다 다름. "
            "baseline 대비 변화의 어느 부분이 데이터 (+swap+bt+hardneg) 때문이고 어느 부분이 "
            "balanced sampler 때문인지는 ablation 한 단계 더 필요.</p>"
        )
    else:
        sections.append(
            "<h3>③ Medium + 전체 증강 + balanced sampler 학습 결과</h3>"
            "<p class='muted'>(학습 진행 중 — 완료되면 자동으로 채워짐. 예상 ~10 시간.)</p>"
        )

    return "\n".join(sections)


def render_summary_table(matrix: dict) -> str:
    header = ["dataset \\ aug"] + AUGS
    rows = []
    for ds in DATASETS:
        row = [f"<b>{ds}</b>"]
        for aug in AUGS:
            c = matrix[ds][aug]
            row.append(
                f"acc {fmt_pct(c['accuracy'])}<br>"
                f"<span class='muted'>F1 {fmt_pct(c['f1'])}<br>"
                f"FPR {fmt_pct(c['fpr'])} · FNR {fmt_pct(c['fnr'])}</span>"
            )
        rows.append(row)
    return render_table(rows, header)


def render_delta_table(matrix: dict) -> str:
    """Show change in accuracy / FPR / FNR vs baseline for each (ds, aug)."""
    header = ["dataset \\ Δ vs baseline"] + [a for a in AUGS if a != "baseline"]
    rows = []
    for ds in DATASETS:
        base = matrix[ds]["baseline"]
        row = [f"<b>{ds}</b>"]
        for aug in AUGS:
            if aug == "baseline":
                continue
            c = matrix[ds][aug]
            d_acc = (c["accuracy"] - base["accuracy"]) * 100
            d_fpr = (c["fpr"] - base["fpr"]) * 100
            d_fnr = (c["fnr"] - base["fnr"]) * 100
            cls = "pos" if d_acc > 0 else ("neg" if d_acc < 0 else "")
            row.append(
                f"<span class='{cls}'>Δacc {d_acc:+.2f}pp</span><br>"
                f"<span class='muted'>ΔFPR {d_fpr:+.2f}pp · ΔFNR {d_fnr:+.2f}pp</span>"
            )
        rows.append(row)
    return render_table(rows, header)


USAGE_HTML = r"""
<h2>새로 추가된 옵션 / 도구 사용법</h2>
<p class="muted">모든 옵션은 기본 off — 명시적으로 켰을 때만 동작. 학습 중인 process 는 이미 메모리에 옛 코드를 로드한 상태라 영향 없음.</p>

<h3>1. <code>--balanced_sampler</code> (학습)</h3>
<p>학습 데이터의 pos/neg 비율이 한쪽으로 쏠려있을 때 epoch 당 50:50 으로 강제. bt 셀의 MRPC FPR 77 % 같은 yes-쏠림 문제 직격.</p>
<p class="muted">위치: main, exp/loss-tuning</p>
<pre>python paraphrase_detection.py --use_gpu \
  --para_train data/paraphrase_extra_data/quora_paws_raw_swap_bt_train.csv \
  --para_dev   data/quora-dev.csv,data/paraphrase_extra_data/paws_dev.csv,data/paraphrase_extra_data/mrpc_traindev.csv \
  --para_dev_out predictions/para-dev-quora-balanced.csv,predictions/para-dev-paws-balanced.csv,predictions/para-dev-mrpc-balanced.csv \
  --para_test data/quora-test-student.csv,data/paraphrase_extra_data/paws_test.csv,data/paraphrase_extra_data/mrpc_test.csv \
  --para_test_out predictions/para-test-quora-balanced.csv,predictions/para-test-paws-balanced.csv,predictions/para-test-mrpc-balanced.csv \
  --model_size gpt2-medium --epochs 10 --batch_size 32 --lr 1e-5 \
  --balanced_sampler</pre>

<h3>2. <code>--symmetric_eval</code> (학습 + 추론)</h3>
<p>dev/test 평가 시 (S1,S2) 와 (S2,S1) 두 방향의 yes/no logit 평균으로 예측. 학습 epoch 평가에도 함께 기록되고, test() 단계에서는 <code>*-symmetric.csv</code> 가 추가로 저장됨. PAWS swap-style FP 직격.</p>
<pre>python paraphrase_detection.py --use_gpu \
  --para_train data/paraphrase_extra_data/quora_paws_raw_swap_bt_train.csv \
  --para_dev   data/paraphrase_extra_data/paws_dev.csv \
  --para_dev_out predictions/para-dev-paws.csv \
  --para_test data/paraphrase_extra_data/paws_test.csv \
  --para_test_out predictions/para-test-paws.csv \
  --model_size gpt2-medium --epochs 10 --batch_size 32 \
  --symmetric_eval
# 산출물: predictions/para-dev-paws.csv  +  predictions/para-dev-paws-symmetric.csv</pre>
<p class="muted"><b>학습 없이 기존 체크포인트로 inference 만</b> 돌리려면 (예: medium 학습 끝난 직후 C3 셀):</p>
<pre>python paraphrase_detection.py --use_gpu \
  --para_dev data/paraphrase_extra_data/paws_dev.csv \
  --para_dev_out predictions/para-dev-paws.csv \
  --para_test data/paraphrase_extra_data/paws_test.csv \
  --para_test_out predictions/para-test-paws.csv \
  --symmetric_eval
# 기본 동작이 train() 후 test() 이므로 train 을 skip 하려면
# paraphrase_detection.py 마지막의 train(args) 호출을 주석 처리하거나
# loss-tuning branch 의 --skip_train flag 패턴을 main 에도 추가하는 게 깔끔</pre>

<h3>3. <code>--swap_augment</code> (학습)</h3>
<p>학습 데이터에 (S2, S1, label) swap 페어 추가 → 데이터 2 배. 순서 대칭성 학습. <code>--symmetric_eval</code> 과 같이 쓰면 학습/추론 양쪽에서 swap-invariance 강화.</p>
<pre>python paraphrase_detection.py --use_gpu \
  --para_train data/paraphrase_extra_data/quora_paws_raw_swap_bt_train.csv \
  --para_dev   data/paraphrase_extra_data/paws_dev.csv \
  --para_dev_out predictions/para-dev-paws.csv \
  --para_test data/paraphrase_extra_data/paws_test.csv \
  --para_test_out predictions/para-test-paws.csv \
  --model_size gpt2-medium --epochs 5 \
  --swap_augment --symmetric_eval</pre>

<h3>4. <code>--prior_calibration</code> (추론 only)</h3>
<p>학습 끝난 모델에 빈 페어 ("", "") 한 번 forward → yes/no logit 의 사전 편향 추정 → 실제 추론 logit 에서 차감. 모델 재학습 없음, 추론 비용 +1 forward.</p>
<pre>python paraphrase_detection.py --use_gpu \
  --para_dev   data/quora-dev.csv,data/paraphrase_extra_data/paws_dev.csv,data/paraphrase_extra_data/mrpc_traindev.csv \
  --para_dev_out predictions/para-dev-quora-prior.csv,predictions/para-dev-paws-prior.csv,predictions/para-dev-mrpc-prior.csv \
  --para_test data/quora-test-student.csv,data/paraphrase_extra_data/paws_test.csv,data/paraphrase_extra_data/mrpc_test.csv \
  --para_test_out predictions/para-test-quora-prior.csv,predictions/para-test-paws-prior.csv,predictions/para-test-mrpc-prior.csv \
  --prior_calibration
# 콘솔에 prior_yes / prior_no / 차이 출력됨 — 양수면 yes 쏠림</pre>

<h3>5. <code>prompts.py</code> — 후보 prompt 모음 (코드 통합 필요)</h3>
<p>6 종 prompt 함수가 정의되어 있음. 현재는 <code>datasets.py</code> 의 <code>collate_fn</code> 이 prompt 를 hardcode 하므로 <b>실제 적용은 다음 한 줄 수정이 필요</b>:</p>
<pre># datasets.py 의 ParaphraseDetectionDataset.collate_fn 내부
from prompts import PROMPTS
template = PROMPTS[getattr(self.p, 'prompt_variant', 'question_pair')]
cloze_style_sents = [template(s1, s2) for (s1, s2) in zip(sent1, sent2)]

# paraphrase_detection.py get_args() 에
parser.add_argument("--prompt_variant", choices=list(PROMPTS.keys()),
                    default='question_pair')</pre>
<p>이후 호출:</p>
<pre>python paraphrase_detection.py --use_gpu --prompt_variant separator ...
python paraphrase_detection.py --use_gpu --prompt_variant nli ...
python paraphrase_detection.py --use_gpu --prompt_variant minimal ...</pre>
<p class="muted">정의된 variant: <code>original</code>, <code>question_pair</code> (현재 학습용 기본), <code>separator</code>, <code>minimal</code>, <code>nli</code>, <code>symmetric</code>.
미리 보려면 <code>python prompts.py</code> 실행.</p>

<h3>6. <code>analysis/build_error_report.py</code> — 사후 분석</h3>
<p>학습/평가가 모두 끝난 다음 prediction csv 만 보고 동작. dev gold 와 매칭, confusion matrix · heuristic 태그별 error rate · 샘플 페어를 모아 이 HTML 을 다시 생성.</p>
<pre>python analysis/build_error_report.py
# → analysis/dev_error_report.html 갱신</pre>
<p>새로운 셀을 추가하고 싶으면 스크립트 상단의:</p>
<pre>DATASETS = ["quora", "paws", "mrpc"]
AUGS     = ["baseline", "raw", "double", "swap", "bt"]</pre>
<p>리스트에 항목만 추가하면 됨 (예: <code>AUGS.append("hardneg")</code>, csv 파일명은 <code>predictions/para-dev-&lt;ds&gt;-&lt;aug&gt;.csv</code> 규칙).</p>

<h3>7. 옵션 조합 빠른 참조</h3>
<table class='grid'>
<thead><tr><th>목적</th><th>옵션 조합</th></tr></thead>
<tbody>
<tr><td>현 학습과 동일 (기본)</td><td>(아무 옵션 안 켜기)</td></tr>
<tr><td>bt-style yes 쏠림 보정</td><td><code>--balanced_sampler</code> (학습) 또는 <code>--prior_calibration</code> (추론)</td></tr>
<tr><td>PAWS swap FP 잡기</td><td><code>--swap_augment --symmetric_eval</code></td></tr>
<tr><td>학습 없이 기존 모델 재평가</td><td><code>--symmetric_eval --prior_calibration</code> (train 호출 주석 처리)</td></tr>
<tr><td>전부 조합</td><td><code>--balanced_sampler --swap_augment --symmetric_eval --prior_calibration</code></td></tr>
</tbody>
</table>
"""


IDEAS_HTML = """
<h2>실제 ablation 수치가 말하는 것</h2>
<p>각 split 의 baseline → raw → double → swap → bt 변화를 보면, 셋이 완전히 다른 이야기를 하고 있다.</p>
<ul>
  <li><b>PAWS</b>: baseline (quora 만 학습) 은 acc 46.7 %, FPR <b>91 %</b> — 거의 모든 페어를 paraphrase 로 찍는다.
      raw 부터 PAWS 가 학습 mix 에 들어오면서 92 % 대로 폭증. swap (92.2 %, FPR 9.3 %) 이 best.
      <i>여기서는 ablation 신호가 매우 강하다.</i></li>
  <li><b>Quora</b>: 89.6 ~ 89.8 % 사이에서 0.2 pp 폭의 미세 변화만. 사실상 평형 — Quora dev 가 saturate.</li>
  <li><b>MRPC</b>: 68.0 ~ 68.7 %. bt 의 acc 가 가장 높지만 FPR 77 %, FNR 9 % — 모델이 거의 모두 "yes" 로 찍어서 얻은 점수다 (precision 하락).
      raw / double / swap 은 FPR 38 ~ 45 % 로 더 균형. 즉 <i>"acc 만 보면 bt 가 1 등" 은 함정</i>.</li>
</ul>
<p>요컨대 <b>학습에 PAWS 가 들어가는가 / 안 들어가는가</b> 외에는 raw vs swap vs double vs bt 간 차이가 노이즈 폭과 비슷하다.
GPT-2 small 의 capacity 가 차이를 만들기 부족했을 가능성이 높고, 동시에 dev set 의 hard slice 가 충분치 않다.</p>

<h2>왜 raw/swap/double/bt 가 비슷해 보이는가</h2>
<ul>
  <li><b>증강이 PAWS positive 위주로 추가</b>. swap 은 전체 49 k 행을 양쪽 모두 추가하므로 PAWS positive : negative 의 원래 비율 (44 % : 56 %) 은 유지되지만 데이터 자체가 PAWS 쪽으로 쏠린다.
      bt 는 label=1 만 21 k 추가 → positive 쪽으로 강하게 skewed → MRPC FPR 폭등의 직접 원인.</li>
  <li><b>모델 capacity / lr 부족</b>. 비슷한 final loss 로 수렴했다면 증강 차이를 흡수할 표현 여유가 없다는 뜻. medium 으로 옮기는 게 정답.</li>
  <li><b>평가 슬라이스가 통째</b>. dev acc 가 평균값이라 PAWS-style swap, numeric mismatch, long sentence 같은 hard slice 의 변화가 묻혀버린다.</li>
</ul>

<h2>오류 분석에서 보이는 공통 약점</h2>
<ol>
  <li><b>PAWS-style word-order swap</b>: jaccard 가 0.85+ 인 페어에서 모델이 두 문장을 거의 같은 의미로 본다. (gold=0, pred=1 의 다수)</li>
  <li><b>수치/엔티티 swap</b>: <i>"…sold to Safeway for $2.5B"</i> ↔ <i>"…for $1.8B"</i> 같은 페어에서 모델이 미세한 차이를 무시한다.</li>
  <li><b>Negation</b>: <i>"does not fit"</i> vs <i>"doesn't fit"</i> 같은 경계 사례 외에, <i>"hated the Iraqi regime"</i> vs <i>"100% behind George Bush"</i> 같은 의미 반전 페어에서도 paraphrase 로 본다.</li>
  <li><b>긴 문장</b>: 길이 200+ MRPC 페어에서 정확도 drop 이 가장 크다 — context length / 위치 정보가 약하다.</li>
  <li><b>Quora 도메인은 question paraphrase</b>이지만 학습은 declarative 위주. wh-word/질문 형식 페어 (특히 <i>"How do I X"</i> vs <i>"What is the best way to X"</i>) 에서 false-negative 가 잦다.</li>
</ol>

<h2>적용 가능한 아이디어 (우선순위)</h2>

<h3>① 학습 신호 자체를 키워라 (성능 균일화 깨기)</h3>
<ul>
  <li><b>Hard-negative augmentation 본격 도입</b>. 이미 <code>paws_train_hardneg.csv</code> (49 k) 가 있는데 ablation 셀에 빠져 있다.
      Cross-pair TF-IDF retrieval 로 만든 hardneg 는 PAWS-style FP 를 직접 잡는다.
      <i>baseline·raw·double·swap·bt·<u>hardneg</u></i> 한 셀을 더 돌리면 FPR delta 가 명확히 보일 가능성이 가장 높다.</li>
  <li><b>Contrastive / triplet loss 추가</b>. (anchor, positive=BT, negative=swap_pair from same scaffolding) triplet 을 만들어
      CE loss 와 0.1~0.3 비중으로 섞으면 ordering-invariance 함정에서 빠져나오기 좋다.</li>
  <li><b>R-Drop / consistency regularization</b>. 같은 페어에 dropout 만 다르게 2회 forward → KL term.
      모델 크기를 medium 으로 키운 지금 같은 시점에 ablation 차이를 키워주는 가장 싼 방법.</li>
</ul>

<h3>② 데이터 분포 자체를 손봐라</h3>
<ul>
  <li><b>Label-balanced sampling</b>. Quora 는 ~63 % negative, PAWS swap/bt 는 positive 쪽으로 쏠리는데 합치면 비율이 흔들린다.
      WeightedRandomSampler 또는 sub-sampling 으로 epoch 당 pos:neg 50:50 으로 맞추기.</li>
  <li><b>Numeric/entity-perturbation augmentation</b>. gold=1 페어에서 숫자/고유명사를 무작위 치환 → label=0 negative 를 자동 생성.
      MRPC 의 가격·날짜·인명 swap 류 FN 을 직격으로 줄인다.</li>
  <li><b>Question-form paraphrase 추가</b>. Quora-quora cross-pair (자기 페어 제외) 에서 wh-prefix mining 으로 paraphrase positives 보강.
      Quora dev 의 5 % 남은 오류 중 절반 이상이 질문 paraphrasing.</li>
  <li><b>MRPC train (3.7 k)</b> 를 학습 mix 에 포함. 지금은 평가 전용으로만 쓰이고 있어서 MRPC dev 가 사실상 OOD.
      OOD 로 보고 싶다면 그렇게 두되, 성능이 목표면 train mix 에 넣는 게 가장 큰 lift.</li>
</ul>

<h3>③ 모델/입력 측면</h3>
<ul>
  <li><b>Prompt 변경</b>. 현재 <code>Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no":</code> 는 length bias 가 크다.
      <code>Sentence A: {s1}\\nSentence B: {s2}\\nDo A and B mean the same? yes / no:</code> 로 separator 를 분리하면
      positional 신호가 깨끗해진다.</li>
  <li><b>Symmetric pairing</b>. 평가 시 (s1,s2)와 (s2,s1) 을 같이 inference → argmax 평균.
      PAWS swap 학습으로 일부 흡수했지만 inference-time 에서도 강제하면 swap-style FP 가 더 잡힌다.</li>
  <li><b>Length-bucketed batching</b>. MRPC 의 긴 페어(>200 토큰) 가 학습 step 당 한두 개씩만 들어가 잘 학습되지 않는다.
      길이 bucket 별 sampler 또는 진짜 long-context 만 모은 mini-epoch 도입.</li>
  <li><b>Auxiliary lexical-overlap feature</b>. last hidden state 옆에 jaccard / length-diff / num-mismatch 4-차원 scalar 를 concat 한
      head 를 둬서 모델이 "표면적 overlap" 신호를 명시적으로 받게 한다. PAWS FP 가 가장 즉각적으로 줄어든다.</li>
</ul>

<h3>④ Ablation 신호를 살리는 평가 디자인</h3>
<ul>
  <li><b>Sliced metrics</b>. 위 분석에서 정의한 <i>word-order-swap / numeric-mismatch / negation / low-overlap / long</i> 5 개 슬라이스로
      나눠서 dev acc 를 매번 찍으면, 평균 acc 가 같아도 어떤 증강이 어떤 slice 를 고치는지 보인다. 매 epoch wandb 로 push.</li>
  <li><b>2 seed × 2 run 표준오차</b>. 현재 ablation 차이가 0.5 pp 이내라면 single-seed 노이즈에 묻힐 수 있다.
      same-config 를 seed 2-3 개로 돌려 변동성 추정 후 비교.</li>
  <li><b>Calibration plot</b>. softmax confidence 와 실제 정답률 비교. 증강마다 모델이 "어떻게 틀리는지" 가 달라지는지 (overconfident-FP vs underconfident-FN) 한 눈에 보임.</li>
</ul>

<h2>현재 실험과 동시 진행 작업 계획</h2>
<p class="muted">학습 중 GPU 포화 (RTX 4090 util 92 %, VRAM 14.3 / 24.5 GB) 상황에서 GPU 자유 없이 진행할 수 있는 것들.
완료된 항목은 <s>취소선</s> 으로 표시.</p>

<h3>트랙 A — CPU 데이터/코드 준비</h3>
<table class='grid'>
<thead><tr><th>#</th><th>작업</th><th>산출물</th><th>비용</th><th>우선</th><th>상태</th></tr></thead>
<tbody>
<tr><td>A1</td><td>Hardneg 학습 셋 mix 확인 (이미 존재하는 <code>quora_paws_raw_swap_bt_hardneg_train.csv</code> 검증 + spec yaml)</td><td>spec yaml</td><td>CPU 5 분</td><td>★★★</td><td>대기</td></tr>
<tr><td>A2</td><td>Numeric/entity perturbation negative 자동 생성 (label=1 페어의 숫자/고유명사 무작위 치환)</td><td><code>*_train_numswap.csv</code></td><td>CPU 20~30 분</td><td>★★★</td><td>대기</td></tr>
<tr><td>A3</td><td><s>Label-balanced sampler — <code>--balanced_sampler</code> flag 추가 (main + exp/loss-tuning)</s></td><td><s>paraphrase_detection.py patch</s></td><td><s>30 분</s></td><td>★★★</td><td><s>완료</s></td></tr>
<tr><td>A4</td><td><s>Symmetric inference wrapper — exp/sentence-swap-symmetry 의 <code>--symmetric_eval</code> / <code>--swap_augment</code> 를 main 으로 통합, branch 삭제</s></td><td><s>evaluation.py + datasets.py + paraphrase_detection.py patch</s></td><td><s>20 분</s></td><td>★★</td><td><s>완료</s></td></tr>
<tr><td>A5</td><td>Question-form (wh) mining — Quora train 에서 wh-prefix 쌍 보강</td><td><code>quora_wh_extra.csv</code></td><td>CPU 15 분</td><td>★</td><td>대기</td></tr>
</tbody>
</table>

<h3>트랙 B — 평가/리포팅 인프라</h3>
<table class='grid'>
<thead><tr><th>#</th><th>작업</th><th>산출물</th><th>비용</th><th>우선</th><th>상태</th></tr></thead>
<tbody>
<tr><td>B1</td><td><s>Error report 코드 — 학습 중이 아닌 결과 csv 만으로 동작 (main only)</s></td><td><s><code>analysis/build_error_report.py</code> (이미 그렇게 작성됨)</s></td><td><s>—</s></td><td>★★★</td><td><s>완료</s></td></tr>
<tr><td>B2</td><td>Calibration plot 스크립트 (softmax conf vs 실제 정답률)</td><td><code>analysis/calibration.py</code></td><td>CPU 20 분</td><td>★★</td><td>대기</td></tr>
<tr><td>B3</td><td>Watcher: 새 prediction csv 생성 시 HTML 자동 재생성</td><td><code>analysis/regen_on_change.sh</code></td><td>CPU 10 분</td><td>★</td><td>대기</td></tr>
<tr><td>B4</td><td><s>후보 prompt 리스트 (main only) — finetuning/inference 단계에 적용</s></td><td><s><code>prompts.py</code> — 6 variants: original, question_pair, separator, minimal, nli, symmetric</s></td><td><s>10 분</s></td><td>★★</td><td><s>완료</s></td></tr>
<tr><td>★</td><td><s>Prior calibration — 빈 문장 dummy inference 로 yes/no logit 편향 추정 → 실제 logit 에서 차감 (main 에 <code>--prior_calibration</code> flag)</s></td><td><s>evaluation.py + paraphrase_detection.py patch</s></td><td><s>30 분</s></td><td>★★★</td><td><s>완료</s></td></tr>
</tbody>
</table>

<h3>트랙 C — 학습 큐 spec (medium 종료 후 직렬 실행)</h3>
<table class='grid'>
<thead><tr><th>#</th><th>셀 이름</th><th>학습 데이터 / 옵션</th><th>기대 검증</th></tr></thead>
<tbody>
<tr><td>C1</td><td><code>medium-hardneg</code></td><td><code>quora_paws_raw_swap_bt_hardneg_train.csv</code> (이미 존재, 452 k)</td><td>PAWS FPR ↓ / MRPC FPR ↓</td></tr>
<tr><td>C2</td><td><code>medium-balanced</code></td><td>C1 + <code>--balanced_sampler</code></td><td>MRPC FPR 77 % (bt 셀) 정상화</td></tr>
<tr><td>C3</td><td><code>medium-symmetric-eval</code></td><td>학습 X, 기존 모델 + <code>--symmetric_eval</code></td><td>PAWS swap-style FP slice 개선폭</td></tr>
<tr><td>C4</td><td><code>medium-numswap</code></td><td>C1 + A2 산출물</td><td>MRPC numeric-mismatch slice 개선</td></tr>
<tr><td>C5</td><td><code>medium-prompt-sep</code></td><td>C1 + prompts.py 의 separator / nli / symmetric variant</td><td>length-bias 감소, prompt 의존도 측정</td></tr>
<tr><td>C6</td><td><code>medium-prior-cal</code></td><td>학습 X, 기존 모델 + <code>--prior_calibration</code></td><td>FPR/FNR 균형, MRPC bt 같이 yes-쏠림 모델 보정</td></tr>
</tbody>
</table>

<h3>권장 진행 순서</h3>
<ol>
  <li><b>지금 즉시 (병렬, CPU)</b>: A1, A2, A5, B2, B3 — 약 1~2 시간</li>
  <li><b>Medium 학습 종료 직후</b>: C3 + C6 (학습 없이 inference 만), 30 분 내</li>
  <li><b>이후 직렬</b>: C1 (3~5 h) → C2 → C4 → C5</li>
</ol>
"""


def main() -> int:
    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    gold_cache: dict[str, dict] = {}

    for ds in DATASETS:
        gold_cache[ds] = load_gold(GOLD_FILE[ds])
        for aug in AUGS:
            p = PRED_DIR / f"para-dev-{ds}-{aug}.csv"
            if not p.exists():
                print(f"missing: {p}", file=sys.stderr)
                matrix[ds][aug] = None
                continue
            preds = load_pred(p)
            matrix[ds][aug] = analyze_cell(gold_cache[ds], preds, ds=ds)

    # EXPERIMENTS × DATASETS 매트릭스 (큰 metric 표용)
    exp_data: list[list[dict | None]] = []
    for exp in EXPERIMENTS:
        row = []
        for ds in DATASETS:
            row.append(load_cell_for_experiment(exp, ds, gold_cache[ds]))
        exp_data.append(row)

    big_table = render_big_metrics_table(exp_data)
    medium_section = render_medium_analysis(exp_data)
    file_list = render_file_listing()

    summary = render_summary_table(matrix)
    deltas = render_delta_table(matrix)

    # small 셀 상세는 접어두고, medium 셀(있는 경우)만 펼침
    small_detail_blocks = []
    for ds in DATASETS:
        for aug in AUGS:
            cell = matrix[ds][aug]
            if cell is None:
                continue
            small_detail_blocks.append(render_cell_block(ds, aug, cell))
    small_detail_html = "\n".join(small_detail_blocks)

    # medium 의 cell-level 상세 (예측 csv 가 있는 medium experiment 모두)
    medium_detail_blocks = []
    for exp_idx, exp in enumerate(EXPERIMENTS):
        if not exp["model"].endswith("medium"):
            continue
        for di, ds in enumerate(DATASETS):
            cell = exp_data[exp_idx][di]
            if cell is None:
                continue
            label = f"{exp['data']} {exp['method']} → {ds}"
            medium_detail_blocks.append(render_cell_block(ds, label, cell))
    medium_detail_html = "\n".join(medium_detail_blocks) or "<p class='muted'>(medium 결과 아직 없음)</p>"

    css = """
    body { font-family: -apple-system, Segoe UI, Helvetica, sans-serif;
           max-width: 1200px; margin: 24px auto; padding: 0 16px;
           color:#111; line-height:1.4; }
    h1 { margin-bottom: 4px; }
    .sub { color:#666; margin-top:0; }
    .tabs { display:flex; gap:4px; border-bottom:2px solid #ddd; margin-top:18px; }
    .tabs button {
        padding:10px 18px; border:none; background:#f3f4f6; cursor:pointer;
        border-radius:8px 8px 0 0; font-size:14px; font-weight:600;
    }
    .tabs button.active { background:#1f2937; color:white; }
    .pane { display:none; padding:20px 4px; }
    .pane.active { display:block; }
    table.grid { border-collapse:collapse; margin: 8px 0; font-size: 13px; }
    table.grid th, table.grid td { border:1px solid #ddd; padding:6px 10px; vertical-align:top; }
    table.grid th { background:#f9fafb; text-align:left; }
    .two-col { display:flex; gap:32px; flex-wrap:wrap; }
    .two-col > div { flex:1; min-width: 320px; }
    .sent { max-width:430px; font-size:12.5px; }
    code { background:#f3f4f6; padding:1px 4px; border-radius:3px; }
    .muted { color:#777; font-size:12px; }
    .tag { font-size:11px; color:#1f2937; background:#fef3c7; padding:1px 6px; border-radius:10px; }
    .pos { color:#047857; font-weight:600; }
    .neg { color:#b91c1c; font-weight:600; }
    details.cell-block { margin: 14px 0; border:1px solid #e5e7eb; border-radius:8px; padding:10px 14px; }
    details.cell-block summary { cursor:pointer; font-size:14px; }
    details.cell-block[open] summary { margin-bottom:10px; }
    h2 { margin-top:28px; }
    h3 { margin-top:18px; color:#1f2937; }
    pre { background:#0f172a; color:#e2e8f0; padding:12px 14px; border-radius:6px;
          overflow-x:auto; font-size:12.5px; line-height:1.5; }
    pre code, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    /* 큰 metric 표 */
    table.big-metrics { font-size:12px; }
    table.big-metrics th.ds-head { text-align:center; background:#e0e7ff; }
    table.big-metrics td.ds-first, table.big-metrics th.ds-first { border-left:2px solid #94a3b8; }
    table.big-metrics tr.group-start td { border-top:2px solid #94a3b8; }
    table.big-metrics td.model-cell { background:#f1f5f9; text-align:center; vertical-align:middle; }
    table.big-metrics td.sticky-l, table.big-metrics th.sticky-l { background:#fafafa; }
    table.file-list { font-size:11.5px; }
    table.file-list tr.muted td { color:#9ca3af; }
    """

    js = """
    function showTab(id) {
        document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        document.getElementById('btn-' + id).classList.add('active');
    }
    """

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Paraphrase ablation — dev error analysis</title>
<style>{css}</style>
</head>
<body>
<h1>Paraphrase ablation — dev error analysis</h1>
<p class="sub">predictions/ × 3 dev split (quora·paws·mrpc) — GPT-2 small (5 ablation) + GPT-2 medium (baseline + inference 옵션 + hardneg-balanced)</p>

<div class="tabs">
  <button id="btn-tab-analysis" class="active" onclick="showTab('tab-analysis')">① Dev 오류 분석</button>
  <button id="btn-tab-ideas" onclick="showTab('tab-ideas')">② 개선 아이디어</button>
  <button id="btn-tab-usage" onclick="showTab('tab-usage')">③ 추가 옵션 사용법</button>
</div>

<div id="tab-analysis" class="pane active">

  <h2>지표 풀이 (먼저 한 번)</h2>
  <ul>
    <li><b>정확도 (acc)</b> = 전체 페어 중 맞춘 비율</li>
    <li><b>F1</b> = 정밀도·재현율 조화평균 (macro)</li>
    <li><b>P (정밀도)</b> = "paraphrase 다" 라고 한 것 중 진짜 paraphrase 비율</li>
    <li><b>R (재현율)</b> = 진짜 paraphrase 중 모델이 맞춘 비율</li>
    <li><b>FPR (오탐률)</b> = 실제로 다른 페어 중 모델이 paraphrase 라고 잘못 본 비율 (↑ = yes 쏠림)</li>
    <li><b>FNR (미탐률)</b> = 실제 paraphrase 중 모델이 놓친 비율 (↑ = no 쏠림)</li>
  </ul>

  <h2>📊 전체 실험 결과 — model × (학습 데이터, 방법) × dev split × metric</h2>
  <p class="muted">가로축: 큰 = dev split (quora/paws/mrpc), 작은 = 6 metric.
  세로축: 큰 = 모델 (small / medium), 작은 = (학습 데이터, 적용 방법).
  결과 csv 가 없는 칸은 "—" (아직 학습/추론 진행 중).</p>
  {big_table}

  <h2>오류 분석</h2>

  <details>
    <summary><b>① GPT-2 small 5 개 ablation 요약 — 클릭해서 펼치기</b>
        <span class='muted'>(이전에 분석한 내용, 요약만 남김)</span></summary>
    <p>핵심만:</p>
    <ul>
      <li><b>PAWS</b>: baseline (quora 만 학습) 은 acc 47 %, FPR 91 % — 거의 모든 페어를 paraphrase 로 찍는다.
          raw 부터 PAWS 가 학습 mix 에 들어오면서 92 % 대로 폭증.
          <i>여기서는 ablation 신호가 매우 강하다.</i></li>
      <li><b>Quora</b>: 89.6 ~ 89.8 % 사이에서 0.2 pp 폭의 미세 변화만. dev 가 saturate.</li>
      <li><b>MRPC</b>: 68.0 ~ 68.7 %. bt 의 acc 가 가장 높지만 FPR 77 %, FNR 9 % — 모델이 거의 모두 "yes" 로 찍어서 얻은 점수.
          <i>"acc 만 보면 bt 가 1 등" 은 함정.</i></li>
    </ul>
    <p><b>데이터셋 3 종 성격 차이</b></p>
    <table class='grid'>
    <thead><tr><th>데이터셋</th><th>출처/성격</th><th>평균 길이</th><th>특징적 신호</th></tr></thead>
    <tbody>
      <tr><td>Quora (40,429)</td><td>사용자 질문쌍, informal·짧음</td><td>~60자</td><td>wh- 시작 88 %, "?" 99.9 %</td></tr>
      <tr><td>PAWS (8,000)</td><td>위키 문장 word-order swap, hard pair</td><td>~113자</td><td>숫자 43 %, 고유명사 헤비 83 %</td></tr>
      <tr><td>MRPC (4,076)</td><td>뉴스 기사, 격식, 인용 다수</td><td>~119자</td><td>인용부호 21 %, 금액·날짜 42 %, 고유명사 62 %</td></tr>
    </tbody>
    </table>
    <p class="muted">small 의 셀 단위 confusion matrix · 태그별 오류율 · 샘플 FP/FN 페어는 아래 "small 셀별 상세" 에 모두 보존.</p>
    <details>
      <summary><b>small 종합 표 (acc/F1/FPR/FNR)</b></summary>
      {summary}
    </details>
    <details>
      <summary><b>small 증강별 Δ vs baseline</b></summary>
      {deltas}
    </details>
    <details>
      <summary><b>small 셀별 상세 (혼동행렬 · 태그별 오류율 · FP/FN 샘플)</b></summary>
      {small_detail_html}
    </details>
  </details>

  <h3>② GPT-2 medium — 학습 결과 + inference 옵션별 효과</h3>
  {medium_section}

  <details>
    <summary><b>medium 셀별 상세 (혼동행렬 · 태그별 오류율 · FP/FN 샘플)</b></summary>
    {medium_detail_html}
  </details>

  <details style='margin-top:30px;'>
    <summary><b>📂 모든 실험의 prediction 파일 경로 목록</b>
      <span class='muted'>(raw 파일 직접 열어보기 / 회색 = 아직 없음)</span></summary>
    <p class="muted">naming 규칙: <code>predictions/para-{{dev|test}}-{{ds}}-{{pattern}}.csv</code>.
    medium 의 -se / -sepc 접미사가 붙은 실험은 <code>-symmetric.csv</code> 가 진짜 적용된 결과 (베이스 .csv 는 비교용).</p>
    {file_list}
  </details>
</div>

<div id="tab-ideas" class="pane">
{IDEAS_HTML}
</div>

<div id="tab-usage" class="pane">
{USAGE_HTML}
</div>

<script>{js}</script>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    # also print a tiny console summary
    for ds in DATASETS:
        for aug in AUGS:
            c = matrix[ds][aug]
            if c is None:
                continue
            print(f"{ds:6s} {aug:9s} acc={c['accuracy']:.4f}  F1={c['f1']:.4f}  FPR={c['fpr']:.4f}  FNR={c['fnr']:.4f}  (n={c['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
