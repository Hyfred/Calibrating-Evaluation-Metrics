from datasets import load_dataset
import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

ds = load_dataset("allenai/openbookqa", "additional")

multiple_choice_answers = ["A", "B", "C", "D"]

dataset = pd.DataFrame(
    {
        "question": ds["train"]["question_stem"],
        "answer": [
            choice["text"][multiple_choice_answers.index(label)]
            for choice, label in zip(ds["train"]["choices"], ds["train"]["answerKey"])
        ],
        "choices": [tuple(choice["text"]) for choice in ds["train"]["choices"]],
    }
)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/openbookqa.csv", index=False)
