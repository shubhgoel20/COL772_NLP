echo "Downloading Model"
# read model url from file "model_url.txt"
MODEL_NAME=$(cat model_url.txt)
# Strip of the trailing newline
MODEL_NAME=${MODEL_NAME%$'\n'}
gdown $MODEL_NAME --fuzzy