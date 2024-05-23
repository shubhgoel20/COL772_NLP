#!/bin/bash

# Define functions for training and inference
train_model() {
    train_file="$1"
    val_file="$2"

    python3 train_col2.py "$train_file" "$val_file"
    python3 train2.py "$train_file"
    
}

run_inference() {
    test_file="$1"
    pred_file="$2"
    
    python3 inference.py "$test_file" "$pred_file"

}

echo "Number of arguments: $#"
echo "Arguments: $@"

if [ "$1" == "test" ]; then
    if [ $# -ne 3 ]; then
        echo "Usage: bash run_model.sh test <test file> <pred file>"
        exit 1
    fi
    run_inference "$2" "$3"
elif [ $# -ne 2 ]; then
    echo "Usage: bash run_model.sh <train file> <val file>"
    exit 1
else
    train_model "$1" "$2"
fi
