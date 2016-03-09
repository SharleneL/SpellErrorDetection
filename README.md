# About folders & files:
1. `keras_detect.py`: the python file to process data & generate 5-fold training and testing datasets.
2. `Data`: contains the category and original/correct/error utterances (each file is one column from Maxine's data)
3. `keras-master`: the keras library

# Running Instructions
1. `cd` to the repository & run `python keras_detect.py`, the python file will generate 5-fold training and testing data under the same folder
2. `cd` to `keras-master/examples` and check the input path in `imdb_lstm.py`(line 40-43), run `python imdb_lstm.py` 