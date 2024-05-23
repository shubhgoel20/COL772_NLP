mode="$1"
case "$mode" in
    train)
        data_path="$2"
        save_path="$3"
        python3 train.py --train_data "$data_path" --save "$save_path" --freq 1
        ;;
    test)
        save_path="$2"
        test_file="$3"
        output_file="$4"
        python3 test.py --test_data "$test_file" --save "$save_path" --output "$output_file"
        ;;
    *)
        echo "Invalid mode. Use 'train' or 'test'."
        exit 1
        ;;
esac