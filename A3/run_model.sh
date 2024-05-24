#!/bin/bash
if [ "$1" == "train" ]; then
    python3 train_script.py "$2" "$3"
elif [ "$1" == "test" ]; then
    python3 inference.py "$2" "$3" "$4"
else
    echo "Invalid command. Use 'train' or 'test'."
fi
