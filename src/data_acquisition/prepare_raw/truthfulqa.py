import json

import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

# From https://github.com/sylinrl/TruthfulQA/blob/main/data/v0/TruthfulQA.csv
file_path = "raw_data/TruthfulQA.csv"

dataset = pd.read_csv(file_path)[["Question", "Best Answer"]]
dataset.rename(columns={"Question": "question", "Best Answer": "answer"}, inplace=True)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/truthfulqa.csv", index=False)
