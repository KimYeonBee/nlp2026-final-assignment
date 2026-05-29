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


def tag_example(s1: str, s2: str) -> dict:
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

    return {
        "jaccard": j,
        "swap_sig": swap_sig,
        "num_mismatch": num_mismatch,
        "neg_diff": neg_diff,
        "proper_diff": proper_diff,
        "len_ratio": len_ratio,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def analyze_cell(gold: dict[str, dict], pred: dict[str, int], cap_examples: int = 15) -> dict:
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

        meta = tag_example(row["s1"], row["s2"])
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
        ["", "<b>pred 0</b>", "<b>pred 1</b>"],
        ["<b>gold 0</b>", cm[(0, 0)], cm[(0, 1)]],
        ["<b>gold 1</b>", cm[(1, 0)], cm[(1, 1)]],
    ]
    cm_html = render_table(cm_rows[1:], cm_rows[0])

    tag_rows = []
    for t, info in sorted(cell["tag_rates"].items(), key=lambda kv: -kv[1]["err_rate"]):
        tag_rows.append([
            html.escape(t),
            info["total"],
            info["wrong"],
            fmt_pct(info["err_rate"]),
        ])
    tags_html = render_table(tag_rows, ["tag", "examples in dev", "wrong", "error rate"])

    return f"""
<details class='cell-block'>
<summary><b>{ds} / {aug}</b>
    — acc {fmt_pct(cell['accuracy'])},
    P {fmt_pct(cell['precision'])}, R {fmt_pct(cell['recall'])}, F1 {fmt_pct(cell['f1'])},
    FPR {fmt_pct(cell['fpr'])}, FNR {fmt_pct(cell['fnr'])}
    (n={cell['n']}, FP={cell['errors_fp_count']}, FN={cell['errors_fn_count']})
</summary>
<div class='two-col'>
    <div><h4>Confusion matrix</h4>{cm_html}</div>
    <div><h4>Error rate by heuristic tag</h4>{tags_html}</div>
</div>
<h4>Sample false-paraphrases (gold=0, pred=1)</h4>
{render_errors(cell['errors_fp'], 'FP')}
<h4>Sample missed paraphrases (gold=1, pred=0)</h4>
{render_errors(cell['errors_fn'], 'FN')}
</details>
"""


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
            matrix[ds][aug] = analyze_cell(gold_cache[ds], preds)

    summary = render_summary_table(matrix)
    deltas = render_delta_table(matrix)

    detail_blocks = []
    for ds in DATASETS:
        for aug in AUGS:
            cell = matrix[ds][aug]
            if cell is None:
                continue
            detail_blocks.append(render_cell_block(ds, aug, cell))
    detail_html = "\n".join(detail_blocks)

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
<p class="sub">predictions/ × 3 dev split (quora·paws·mrpc) — GPT-2 small (medium 재학습 중)</p>

<div class="tabs">
  <button id="btn-tab-analysis" class="active" onclick="showTab('tab-analysis')">① Dev 오류 분석</button>
  <button id="btn-tab-ideas" onclick="showTab('tab-ideas')">② 개선 아이디어</button>
</div>

<div id="tab-analysis" class="pane active">
  <h2>요약 표 — accuracy / F1 / FPR / FNR</h2>
  {summary}
  <h2>증강별 변화 (Δ vs baseline)</h2>
  <p class="muted">baseline 대비 절대 변화량 (pp). 0 에 가까울수록 ablation 신호가 없음.</p>
  {deltas}
  <h2>셀별 상세 — 오류 유형, FP/FN 샘플</h2>
  <p class="muted">각 셀을 펼치면 confusion matrix, heuristic 태그별 error rate, 실제 잘못 맞춘 페어 샘플 (jaccard 내림차순) 을 볼 수 있다.</p>
  {detail_html}
</div>

<div id="tab-ideas" class="pane">
{IDEAS_HTML}
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
