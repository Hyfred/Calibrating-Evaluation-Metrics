import math
import os
import pickle
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
# from src.utils import read_tables
from transformers import pipeline
from tqdm import tqdm 

def chunk_text_by_char(text, max_length):
    # Split the text into chunks of fixed character length
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def embed(pipeline, input):
    # Tokenize and split into chunks
    chunks = chunk_text_by_char(input, 512)

    # Process each chunk
    results = []
    for chunk in chunks:
        # cls embedding
        result = pipeline(chunk, return_tensors="pt")[0].numpy()[0]
        results.append(result)

    return np.mean(results, axis=0)


@hydra.main(
    version_base=None,
    config_path="../../configs/data_acquisition",
    config_name="embeddings",
)
def main(cfg: DictConfig) -> None:
    ds = pd.read_csv(
        f"processed_data/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}.csv"
    )
    ds = ds[ds[["question", "llm_answer"]].notna().all(axis=1)]

    ds["question"] = ds["question"].astype("string")
    ds["llm_answer"] = ds["llm_answer"].astype("string")
    if "choices" in ds.columns:
        ds["choices"] = ds["choices"].apply(eval)

    pipe = pipeline(
        "feature-extraction",
        model="distilbert-base-uncased",
        device=cfg.device,
        max_length=512,
        truncation=True,
    )

    embs = {}
    for idx, row in tqdm(ds.iterrows(), total=len(ds)):
        try:
            question_emb = embed(pipe, row["question"])
            answer_emb = embed(pipe, row["llm_answer"])
            embs[(row["question"], row["llm_answer"])] = np.mean(
                [question_emb, answer_emb], axis=0
            )
        except:
            import pdb

            pdb.set_trace()
            pass

    with open(
        f"processed_data/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}_embeddings.pickle",
        "wb",
    ) as fh:
        pickle.dump(embs, fh)


if __name__ == "__main__":
    main()