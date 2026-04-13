import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from src.data_acquisition.generation import (
    answers_are_equivalent_llm,
    get_onestage_linguistic_topk_guess_fn,
)
from src.data_acquisition.parsing import (
    CONFIDENCE_EXPRESSIONS,
    CONFIDENCE_EXPRESSIONS_PROBABILITIES,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/data_acquisition",
    config_name="judgement",
)
def main(cfg: DictConfig) -> None:

    ds = pd.read_csv(
        f"processed_data/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}.csv"
    )
    ds["answer"] = ds["answer"].astype("string")
    ds["llm_answer"] = ds["llm_answer"].astype("string")
    if "choices" in ds.columns:
        ds["choices"] = ds["choices"].apply(eval)

    new_rows = []
    for _, row in ds.iterrows():

        try:
            if "choices" in row:
                gt, raw_response = answers_are_equivalent_llm(
                    row["question"],
                    row["answer"].lower(),
                    row["llm_answer"].lower(),
                    choices=row["choices"],
                    model=cfg.judgement_llm,
                    verbose=False,
                )
            else:
                gt, raw_response = answers_are_equivalent_llm(
                    row["question"],
                    row["answer"].lower(),
                    row["llm_answer"].lower(),
                    model=cfg.judgement_llm,
                    verbose=False,
                )

            gt = float(gt)
        except:
            gt = None
            raw_response = None

        row["judgement_value"] = gt
        row["judgement_value_raw"] = raw_response

        print(row)
        new_rows.append(row)

    ds_judgement_output = pd.DataFrame(new_rows)
    pd.DataFrame(ds_judgement_output).to_csv(
        f"processed_data/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}_{cfg.judgement_llm}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
