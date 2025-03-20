import pandas as pd

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(['Unnamed: 0', 'machine_answer'], axis=1)
    return data

def split_data(data, train_ratio=0.8):
    split_index = int(train_ratio * len(data))
    train = data[:split_index]
    test = data[split_index:]
    return train, test
