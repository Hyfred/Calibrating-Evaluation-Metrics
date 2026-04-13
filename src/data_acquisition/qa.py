import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from src.data_acquisition.generation import (
    LINGUISTIC_PROMPTS,
    get_onestage_linguistic_topk_guess_fn,
    get_onestage_verbalize_topk_guess_fn,
)
from src.data_acquisition.parsing import (
    CONFIDENCE_EXPRESSIONS,
    CONFIDENCE_EXPRESSIONS_PROBABILITIES,
)


@hydra.main(
    version_base=None, config_path="../../configs/data_acquisition", config_name="qa"
)
def main(cfg: DictConfig) -> None:

    # generate new_answers using the following prompt function
    # prompt_name = "just_ask:ling1s-topk"
    prompt_name = cfg.prompt_name
    if prompt_name == "ling1s-topk":
        _prompt_fn = get_onestage_linguistic_topk_guess_fn(
            k=1, system_prompt=None, model=cfg.qa_llm, n=1, debug=False
        )
    elif prompt_name == "verb1s-topk":
        _prompt_fn = get_onestage_verbalize_topk_guess_fn(
            k=1, system_prompt=None, model=cfg.qa_llm, n=1, debug=False
        )
    else:
        raise NotImplementedError()

    dataset_name = cfg.dataset_name
    ds = pd.read_csv(f"processed_data/{dataset_name}.csv")

    if "choices" in ds.columns:
        ds["choices"] = ds["choices"].apply(eval)

    # downsample ds up to cfg.max_no_samples rows
    if cfg.max_no_samples is not None and len(ds) > cfg.max_no_samples:
        ds = ds.sample(n=cfg.max_no_samples, random_state=cfg.seed)
    else:
        ds = ds.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    new_rows = []
    for _, row in ds.iterrows():
        # if column choice is in row, then we need to pass it to the prompt function
        if "choices" in row:
            llm_answer, llm_confidence = _prompt_fn(row["question"], row["choices"])
        else:
            llm_answer, llm_confidence = _prompt_fn(row["question"])

        if isinstance(llm_answer, list):
            llm_answer = llm_answer[0]
        if isinstance(llm_confidence, list):
            llm_confidence = llm_confidence[0]

        raw_confidence = None
        if prompt_name in LINGUISTIC_PROMPTS:
            numerical_confidence = None
            for confidence_idx, confidence_expression in enumerate(
                CONFIDENCE_EXPRESSIONS
            ):
                if confidence_expression.lower() == llm_confidence:
                    numerical_confidence = CONFIDENCE_EXPRESSIONS_PROBABILITIES[
                        confidence_idx
                    ]
                    break
            raw_confidence = llm_confidence
            llm_confidence = numerical_confidence

        row["llm_answer"] = llm_answer
        row["llm_confidence"] = llm_confidence
        row["raw_confidence"] = raw_confidence

        print(row)
        new_rows.append(row)

    ds_qa_output = pd.DataFrame(new_rows)
    pd.DataFrame(ds_qa_output).to_csv(
        f"processed_data/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
