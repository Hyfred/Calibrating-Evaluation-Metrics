import json
import re

import pandas as pd

from src.data_acquisition.prepare_raw.utils import check_schema_and_cleanup

file_path = "raw_data/MathQA/train.json"

with open(file_path, "r") as file:
    raw = json.load(file)

options_pattern = r"[a-e] \) (?:[a-e] \) )?([^,]+)"


def extract_answer(options_str, correct_str):
    # Use regex to find all the options
    options = re.findall(r"([a-e]) \) (?:[a-e] \) )?([^,]+)", options_str)
    
    options_dict = {match[0]: match[1].strip() for match in options}
    return options_dict[correct_str]


def extract_choices(options_str):
    # Use regex to find all the options
    options = re.findall(r"[a-e] \) (?:[a-e] \) )?([^,]+)", options_str)

    return options


dataset = pd.DataFrame(
    [
        {
            "question": instance["Problem"],
            "answer": str(extract_answer(instance["options"], instance["correct"])),
            "choices": tuple(extract_choices(instance["options"])),
        }
        for instance in raw
    ]
)

dataset = check_schema_and_cleanup(dataset)

dataset.to_csv("processed_data/mathqa.csv", index=False)
