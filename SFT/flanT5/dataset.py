from datasets import Dataset, DatasetDict

def create_dataset(train, test):
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset
