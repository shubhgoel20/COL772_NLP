import json
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def combine_json(json_files):
    combined_data = []
    for file_path in json_files:
        data = load_json(file_path)
        combined_data.extend(data)
    return combined_data

def split_train_test(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def main():
    # List of paths to your JSON files
    json_files = ['./a1_data/train.json', './a1_data/valid.json', './a1_data/valid_new.json']

    # Combine JSON data from multiple files
    combined_data = combine_json(json_files)

    # Split combined data into train and test sets
    train_data, test_data = split_train_test(combined_data)

    # Save train and test data to separate JSON files
    save_json(train_data, './a1_data2/train.json')
    save_json(test_data, './a1_data2/test.json')

if __name__ == "__main__":
    main()
