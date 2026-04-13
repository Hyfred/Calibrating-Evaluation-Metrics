import json
import re

import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

# unfiltered dataset with 110K question-answer pairs
file_path = "raw_data/triviaqa-unfiltered/unfiltered-web-dev.json"

with open(file_path, "r") as file:
    raw = json.load(file)

dataset = pd.DataFrame(
    [
        {
            "question": instance["Question"],
            "answer": instance["Answer"]["Value"],
        }
        for instance in raw["Data"]
    ]
)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/triviaqa.csv", index=False)
