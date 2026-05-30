# augment_paraphrase.py
#
# PAWS train 증강 파이프라인:
#   1) swap     - S1/S2 순서 뒤집기 (paraphrase 대칭성 부여)
#   2) bt       - en->de->en back-translation (label=1 페어 대상)
#   3) hardneg  - cross-pair TF-IDF retrieval (어휘 유사 / 의미 다른 페어)
#   4) merge_train - Quora + PAWS 변형들을 합쳐 최종 학습 파일 생성
#
# 산출물 (data/paraphrase_extra_data/):
#   - paws_train_swap.csv
#   - paws_train_bt.csv
#   - paws_train_hardneg.csv
#   - quora_paws_<parts>_train.csv  (예: quora_paws_raw_swap_bt_hardneg_train.csv)
#
# 사용 예:
#   python augment_paraphrase.py --steps swap                  # swap 만
#   python augment_paraphrase.py --steps bt                    # BT 만 (GPU 권장, ~30~60분)
#   python augment_paraphrase.py --steps hardneg               # hardneg 만
#   python augment_paraphrase.py --steps merge_train \
#       --merge_parts raw swap                                 # 병합본 생성
#   python augment_paraphrase.py --steps swap bt hardneg merge_train \
#       --merge_parts raw swap bt hardneg                      # 한 번에 전체

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent
EXTRA = DATA_DIR / "paraphrase_extra_data"
EXTRA.mkdir(exist_ok=True)

WS = re.compile(r"\s+")


def _norm(s):
    if not isinstance(s, str):
        return ""
    return WS.sub(" ", s.strip().lower())


def _load_tsv(path):
    return pd.read_csv(
        path, sep="\t", encoding="utf-8-sig",
        dtype={"id": str}, keep_default_na=False
    )


def _save_tsv(df, path):
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"Saved: {path} | rows = {len(df)}")


# ---------------------------------------------------------------------------
# 1) S1/S2 swap
# ---------------------------------------------------------------------------

def augment_swap():
    """모든 PAWS train 행을 (s2, s1, label) 로 뒤집어 추가."""
    df = _load_tsv(EXTRA / "paws_train.csv")
    swap = df.copy()
    swap["id"] = swap["id"].apply(lambda i: f"{i}_swap")
    swap["sentence1"], swap["sentence2"] = df["sentence2"].values, df["sentence1"].values
    _save_tsv(swap, EXTRA / "paws_train_swap.csv")


# ---------------------------------------------------------------------------
# 2) Back-translation (en -> de -> en)
# ---------------------------------------------------------------------------

def augment_bt(
    model_fwd="Helsinki-NLP/opus-mt-en-de",
    model_bwd="Helsinki-NLP/opus-mt-de-en",
    batch_size=64,
    max_length=128,
    num_beams=2,
    device=None,
):
    """label=1 페어의 sentence2 만 BT 변환 -> 새로운 (s1, s2_bt, 1) 페어 생성."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BT device = {device}")

    df = _load_tsv(EXTRA / "paws_train.csv")
    pos = df[df["is_duplicate"].astype(float) == 1.0].reset_index(drop=True)
    print(f"BT 대상 (label=1) = {len(pos)} pairs")

    def _translate(texts, tok, mdl, desc):
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            chunk = list(texts[i:i + batch_size])
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                gen = mdl.generate(**inputs, max_length=max_length * 2, num_beams=num_beams)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    print(f"Loading {model_fwd} ...")
    tok_fwd = MarianTokenizer.from_pretrained(model_fwd)
    mdl_fwd = MarianMTModel.from_pretrained(model_fwd).to(device).eval()

    print(f"Loading {model_bwd} ...")
    tok_bwd = MarianTokenizer.from_pretrained(model_bwd)
    mdl_bwd = MarianMTModel.from_pretrained(model_bwd).to(device).eval()

    src = pos["sentence2"].tolist()
    mid = _translate(src, tok_fwd, mdl_fwd, "en->de")
    bt = _translate(mid, tok_bwd, mdl_bwd, "de->en")

    aug = pos.copy()
    aug["sentence2"] = bt
    aug["id"] = aug["id"].apply(lambda i: f"{i}_bt")

    # 품질 필터: 원문과 동일 / 비정상 길이 비율 제외
    def _keep(orig, new):
        if not isinstance(new, str) or len(new.strip()) == 0:
            return False
        o, n = _norm(orig), _norm(new)
        if o == n:
            return False
        ratio = len(n) / max(len(o), 1)
        if ratio < 0.5 or ratio > 2.0:
            return False
        return True

    mask = [_keep(o, n) for o, n in zip(src, bt)]
    kept = aug[mask].reset_index(drop=True)
    print(f"BT 필터 통과 = {len(kept)} / {len(aug)}")
    _save_tsv(kept, EXTRA / "paws_train_bt.csv")


# ---------------------------------------------------------------------------
# 3) Cross-pair hard-negative retrieval
# ---------------------------------------------------------------------------

def augment_hardneg(topk_candidates=20, jaccard_min=0.5, per_anchor=1, batch_size=1000):
    """
    PAWS train 의 모든 sentence 를 TF-IDF 인덱싱.
    각 페어의 sentence1 을 anchor 로 두고, 다른 페어에서 어휘 유사도가 높은
    문장을 retrieve -> (s1, retrieved, 0) 새 hardneg 페어 생성.

    필터:
      - 같은 페어 내 문장은 제외 (자기 자신/자기 paraphrase 회피)
      - Jaccard 토큰 overlap >= jaccard_min
      - 페어당 per_anchor 개만 채택 (인플레이션 방지)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = _load_tsv(EXTRA / "paws_train.csv")

    # 모집단: PAWS 의 모든 sentence (s1 + s2), pair_id 와 함께 보관
    pool = pd.concat([
        df[["id", "sentence1"]].rename(columns={"sentence1": "text"}),
        df[["id", "sentence2"]].rename(columns={"sentence2": "text"}),
    ], ignore_index=True)
    pool["text_norm"] = pool["text"].map(_norm)

    print(f"TF-IDF 인덱싱 (코퍼스 {len(pool)} 문장) ...")
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    X_pool = vec.fit_transform(pool["text_norm"].tolist())

    anchors_text = df["sentence1"].map(_norm).tolist()
    anchors_id = df["id"].tolist()
    X_anchor = vec.transform(anchors_text)

    pool_ids = pool["id"].tolist()
    pool_texts = pool["text"].tolist()
    pool_norms = pool["text_norm"].tolist()

    def _tokens(s):
        return set(s.split())

    # Cosine similarity 를 anchor 배치로 잘라서 계산 (전체 dense 행렬은 ~20GB+ 로 OOM).
    # 청크당 dense 사이즈 = batch_size × N_pool × 4byte 정도.
    n_anchor = X_anchor.shape[0]
    print(f"Cosine similarity (batched, batch_size={batch_size}, n_anchor={n_anchor}) ...")
    XPT = X_pool.T.tocsr()

    out_rows = []
    skipped = 0
    for start in tqdm(range(0, n_anchor, batch_size), desc="hardneg retrieval"):
        end = min(start + batch_size, n_anchor)
        # 청크 단위로만 dense 화 -> 메모리 OK
        sims_chunk = (X_anchor[start:end] @ XPT).toarray()
        for k, i in enumerate(range(start, end)):
            a_tok = _tokens(anchors_text[i])
            if len(a_tok) == 0:
                skipped += 1
                continue
            # 청크 1행에 대해서만 argsort -> top-K 후보
            order = sims_chunk[k].argsort()[::-1]
            pair_id = anchors_id[i]
            found = 0
            for j in order[:topk_candidates]:
                if pool_ids[j] == pair_id:
                    continue
                c_tok = _tokens(pool_norms[j])
                if len(c_tok) == 0:
                    continue
                jacc = len(a_tok & c_tok) / len(a_tok | c_tok)
                if jacc < jaccard_min:
                    continue
                out_rows.append({
                    "id": f"{pair_id}_hardneg{found}",
                    "sentence1": df.iloc[i]["sentence1"],
                    "sentence2": pool_texts[j],
                    "is_duplicate": 0.0,
                })
                found += 1
                if found >= per_anchor:
                    break
    print(f"hardneg 생성 = {len(out_rows)} (anchor 빈문장 skip = {skipped})")
    aug = pd.DataFrame(out_rows)
    _save_tsv(aug, EXTRA / "paws_train_hardneg.csv")


# ---------------------------------------------------------------------------
# 4) 최종 학습 파일 병합
# ---------------------------------------------------------------------------

def build_merged_train(parts):
    """
    Quora train + 선택된 PAWS 변형들을 concat.
    parts 는 ['raw', 'swap', 'bt', 'hardneg'] 중 부분집합.
    """
    quora = _load_tsv(DATA_DIR / "quora-train.csv")
    dfs = [quora]
    if "raw" in parts:
        dfs.append(_load_tsv(EXTRA / "paws_train.csv"))
    for p in ("swap", "bt", "hardneg"):
        if p in parts:
            f = EXTRA / f"paws_train_{p}.csv"
            if not f.exists():
                raise FileNotFoundError(f"{f} 가 없습니다. 먼저 augment_{p} 단계를 실행하세요.")
            dfs.append(_load_tsv(f))
    merged = pd.concat(dfs, ignore_index=True)
    suffix = "_".join(parts) if parts else "raw"
    out = EXTRA / f"quora_paws_{suffix}_train.csv"
    _save_tsv(merged, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", nargs="+",
        choices=["swap", "bt", "hardneg", "merge_train"],
        default=["swap", "bt", "hardneg", "merge_train"],
        help="실행할 증강 단계",
    )
    parser.add_argument(
        "--merge_parts", nargs="+",
        choices=["raw", "swap", "bt", "hardneg"],
        default=["raw", "swap", "bt", "hardneg"],
        help="merge_train 단계에서 합칠 부분집합",
    )
    args = parser.parse_args()

    if "swap" in args.steps:
        augment_swap()
    if "bt" in args.steps:
        augment_bt()
    if "hardneg" in args.steps:
        augment_hardneg()
    if "merge_train" in args.steps:
        build_merged_train(args.merge_parts)


if __name__ == "__main__":
    main()
