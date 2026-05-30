# make_paraphrase_datasets.py

# 필요 패키지:
# pip install datasets pandas

from datasets import load_dataset
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "paraphrase_extra_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_tsv(df, save_path):
    """
    Quora 데이터셋처럼 tab으로 구분된 csv/tsv 파일로 저장.
    """
    df.to_csv(
        save_path,
        sep="\t",
        index=False,
        encoding="utf-8"
    )
    print(f"Saved: {save_path} | rows = {len(df)}")


def convert_mrpc():
    """
    MRPC dataset
    Hugging Face: load_dataset("glue", "mrpc")

    Columns:
    - sentence1
    - sentence2
    - label
    - idx
    """

    print("Downloading MRPC...")

    dataset = load_dataset("glue", "mrpc")

    split_name_map = {
        "train": "train",
        "validation": "dev",
        "test": "test"
    }

    for hf_split in split_name_map:
        save_split = split_name_map[hf_split]

        df = dataset[hf_split].to_pandas()

        # Quora 형식 (id, sentence1, sentence2, is_duplicate) 으로 변환
        df = df.rename(columns={"label": "is_duplicate"})
        df["id"] = df["idx"].apply(lambda i: f"mrpc_{i}")
        df["is_duplicate"] = df["is_duplicate"].astype(float)
        df = df[["id", "sentence1", "sentence2", "is_duplicate"]]

        save_path = OUTPUT_DIR / f"mrpc_{save_split}.csv"
        save_tsv(df, save_path)


def convert_paws():
    """
    PAWS dataset
    Hugging Face: load_dataset("paws", "labeled_final")

    Columns:
    - id
    - sentence1
    - sentence2
    - label
    """

    print("Downloading PAWS...")

    try:
        dataset = load_dataset("paws", "labeled_final")
    except:
        # 일부 환경에서는 이 이름으로 동작할 수 있음
        dataset = load_dataset("google-research-datasets/paws", "labeled_final")

    split_name_map = {
        "train": "train",
        "validation": "dev",
        "test": "test"
    }

    for hf_split in split_name_map:
        save_split = split_name_map[hf_split]

        df = dataset[hf_split].to_pandas()

        # Quora 형식 (id, sentence1, sentence2, is_duplicate) 으로 변환
        df = df.rename(columns={"label": "is_duplicate"})
        df["id"] = df["id"].apply(lambda i: f"paws_{i}")
        df["is_duplicate"] = df["is_duplicate"].astype(float)
        df = df[["id", "sentence1", "sentence2", "is_duplicate"]]

        save_path = OUTPUT_DIR / f"paws_{save_split}.csv"
        save_tsv(df, save_path)


def merge_train_dev_data():
    """
    선택사항:
    MRPC + PAWS를 합친 train/dev 파일도 생성.
    테스트셋은 보통 최종 평가용이므로 여기서는 합치지 않음.
    """

    print("Merging train/dev data...")

    mrpc_train = pd.read_csv(OUTPUT_DIR / "mrpc_train.csv", sep="\t")
    mrpc_dev = pd.read_csv(OUTPUT_DIR / "mrpc_dev.csv", sep="\t")

    paws_train = pd.read_csv(OUTPUT_DIR / "paws_train.csv", sep="\t")
    paws_dev = pd.read_csv(OUTPUT_DIR / "paws_dev.csv", sep="\t")

    merged_train = pd.concat(
        [mrpc_train, paws_train],
        axis=0,
        ignore_index=True
    )

    merged_dev = pd.concat(
        [mrpc_dev, paws_dev],
        axis=0,
        ignore_index=True
    )

    # 혹시 모를 중복 제거
    merged_train = merged_train.drop_duplicates(
        subset=["sentence1", "sentence2", "is_duplicate"]
    ).reset_index(drop=True)

    merged_dev = merged_dev.drop_duplicates(
        subset=["sentence1", "sentence2", "is_duplicate"]
    ).reset_index(drop=True)

    save_tsv(merged_train, OUTPUT_DIR / "mrpc_paws_train.csv")
    save_tsv(merged_dev, OUTPUT_DIR / "mrpc_paws_dev.csv")


def merge_mrpc_traindev():
    """
    MRPC train + dev 를 한 파일로 합쳐 OOD dev 용 세트 생성.
    이 파일은 학습에 사용하지 않고 dev 평가 전용.
    """

    print("Merging MRPC train + dev for OOD dev set...")

    train = pd.read_csv(OUTPUT_DIR / "mrpc_train.csv", sep="\t", encoding="utf-8-sig")
    dev = pd.read_csv(OUTPUT_DIR / "mrpc_dev.csv", sep="\t", encoding="utf-8-sig")

    merged = pd.concat([train, dev], axis=0, ignore_index=True)
    save_tsv(merged, OUTPUT_DIR / "mrpc_traindev.csv")


def check_files():
    """
    저장 결과 확인용.
    """

    print("\nChecking saved files...\n")

    file_names = [
        "mrpc_train.csv",
        "mrpc_dev.csv",
        "mrpc_test.csv",
        "paws_train.csv",
        "paws_dev.csv",
        "paws_test.csv",
        "mrpc_paws_train.csv",
        "mrpc_paws_dev.csv",
        "mrpc_traindev.csv"
    ]

    for file_name in file_names:
        path = OUTPUT_DIR / file_name

        if path.exists():
            df = pd.read_csv(path, sep="\t")
            print("=" * 80)
            print(file_name)
            print(df.shape)
            print(df.head())
            print(df["is_duplicate"].value_counts())
            print()


def main():
    # convert_mrpc()
    # convert_paws()

    # 필요 없으면 이 줄은 주석 처리해도 됨
    # merge_train_dev_data()
    merge_mrpc_traindev()

    # check_files()


if __name__ == "__main__":
    main()