def check_schema_and_cleanup(dataset):
    # Check if the DataFrame has the required columns
    required_columns = ["question", "answer"]
    assert all(
        column in dataset.columns for column in required_columns
    ), "DataFrame must have 'question', 'answer' columns."

    # Check if there are any null values in the 'question' and 'answer' columns
    assert (
        dataset[["question", "answer"]].notnull().all().all()
    ), "Columns 'question' and 'answer' must not contain null values."

    dataset.drop_duplicates(inplace=True)

    """
    (Pdb) dataset[dataset["question"] == "The language of Prison Break is"]
                                question   answer
    14335  The language of Prison Break is  English
    14336  The language of Prison Break is   Arabic
    """

    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Deduplicate based on the 'question' column, keeping a random occurrence
    dataset = dataset.drop_duplicates(subset="question")

    print(dataset.describe())
    return dataset
