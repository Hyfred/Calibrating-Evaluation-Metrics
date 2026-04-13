import json

import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

# From https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/qa_wikidata/task.json
file_path = "raw_data/bigbench/task.json"

with open(file_path, "r") as file:
    raw = json.load(file)

dataset = pd.DataFrame(raw["examples"])
dataset.rename(columns={"input": "question", "target": "answer"}, inplace=True)


# Define a function to pick the first element if the value is a list
def pick_first_element(val):
    if isinstance(val, list):  # Check if the value is a list
        return val[0]  # Return the first element
    return val  # Return the value as it is if not a list


# Apply the function to all columns of the DataFrame
dataset = dataset.map(pick_first_element)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/bigbench.csv", index=False)
