from datasets import load_dataset
import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

ds = load_dataset("allenai/sciq")

dataset = pd.DataFrame(
    [
        {"question": question, "answer": answer}
        for question, answer in zip(
            ds["train"]["question"], ds["train"]["correct_answer"]
        )
    ]
)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/sciq.csv", index=False)
