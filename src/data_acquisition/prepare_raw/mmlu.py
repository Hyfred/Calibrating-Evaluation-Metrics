from datasets import load_dataset
import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

ds = load_dataset("cais/mmlu", "all")

dataset = pd.DataFrame(
    {
        "question": ds["test"]["question"],
        "answer": [
            str(choices[answer])
            for choices, answer in zip(ds["test"]["choices"], ds["test"]["answer"])
        ],
        "choices": [tuple(choice) for choice in ds["test"]["choices"]],
    }
)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/mmlu.csv", index=False)
