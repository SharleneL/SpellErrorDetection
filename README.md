# About folders & files:
1. `Code`: contains the code & lib
2. `Data`: contains the category and original/correct/error utterances (each file is one column from Maxine's data)

# Running Instructions
## To run deep learning algorithm
1. `cd` to the `Code` folder & run `python keras_detect.py`, the python file will generate 5-fold training and testing data under `train_test_datasets` folder
2. `cd` to `Code` folder and check the input path in `DataForm.py`; run `python lstm.py` 

## To run sk-learn logistic regression algorithm
1. `cd` to the `Code` folder & open `sklearn_lr_detect.py` file
2. Change the input file path in line 30~34, if needed; change the parameter of `get_utterances` function in line 42, 48, 53, if needed
3. Run `python sklearn_lr_detect.py`