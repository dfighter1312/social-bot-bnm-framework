# Supervised Learning for Social Bot Detection - Benchmarking Framework

## To run the framework

- Clone the repository.
- Asking the creators of MIB or TwiBot-20 dataset (Due to the terms of use, we cannot public the dataset onto this site. Dataset requests for academic purpose are adrently accepted. Futher information can be found at: [MIB Dataset](http://mib.projects.iit.cnr.it/dataset.html) and [TwiBot-20 dataset](wind_binteng@stu.xjtu.edu.cn)).
- In `main.py`, uncomment a line for declaring a pipeline and a line for running that pipeline. For example:

```python
if __name__ == "__main__":
  pipeline = BasicPipeline()
  pipeline.run(dataset_name="MIB")
```

- Check the evaluation

```
Dataset name: MIB (3892 samples)
Accuracy: 0.9645
Precision: 0.9711
Recall: 0.8911
MCC: 0.9073
Training time 0.1681s
Inference time: 0.0331s
```

## To implement a new detector

- Check the [list of features](https://drive.google.com/file/d/1DWuNymZxqCEF8v7ySHTLOAU3EIhIlvHp/view?usp=sharing)
- Inherent your detector from `BaseDetectorPipeline`. For example:

```python
class NewPipeline(BaseDetectorPipeline):
  pass
```

- Initialize the list of used features and level of detection via below parameters:

| Parameter | Type | Description |
| :--- | :---: | :--- |
| user_features | List<str>, None or `all` | List of user property features that are employed, they will be included in `user_df` (except the ones that do not exist in the dataset). If `all`, all available features will be used. |
| tweet_metadata_features | List<str>, None or `all` |  List of tweet metadata property features that are employed, they will be included in `tweet_metadata_df` (except the ones that do not exist in the dataset). If `all`, all available features will be used. |
| use_tweet | bool | Decide to use tweet semantic or not. If True, `tweet_df` will return a Dataframe. Otherwise, `tweet_df` will be None. |
| use_network | bool | Decide to use tweet semantic or not. If True, `network_df` will return a Dataframe. Otherwise, `network_df` will be None. Force to be False if using MIB dataset. |
| verbose | bool | If True, print out the process to console. |
| account_level | bool | Clarify the classifier is account-level or not (account-level detector is such detector that making predictions on each account). |
  
- Override some functions:
  
| Function | Arguments | Optional | Description |
| :--- | :---: | :--: | :--- |
| __init__ | Any | | Initialization and run `__init__()` of `BaseDetectorPipeline`. See examples for more details. |
| feature_engineering_u | user_df: Dataframe or None, training: bool | :white_check_mark: | Process feature engineering on user property dataframe. Returned dataframe must include label and user_id column if concatenate function is not overridden. The training parameter indicates the dataset is in training set (false if in validation and test set). |
| feature_engineering_ts | tweet_metadata_df: Dataframe or None, training: bool | :white_check_mark: | Process feature engineering on tweet metadata dataframe. Returned dataframe must include label and user_id column if concatenate function is not overridden. |
| feature_engineering_n | network_df: Dataframe or None, training: bool | :white_check_mark: | Process feature engineering on network dataframe. Returned dataframe must include label and user_id column if concatenate function is not overridden. *Currently not in use* |
| semantic_encoding | tweet_df: Dataframe or None, training: bool | :white_check_mark: | Encode the text on tweet dataframe. Returned dataframe must include label and user_id column if concatenate function is not overridden. |
| concatenate | All dataframes | :white_check_mark: | Concatenate all dataframes into a single dataframe. Don't need to be overriden if only one dataframe is not None. |
| classify | X_train, X_dev, y_train, y_dev |  | Fit the data into a model. **Must be implemented.** |
| predict | X_test | | Return the predicted result from the test data. **Must be implemented.** |
  
*You can check some examples in the `src/example` folder.*
