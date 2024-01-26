# This file is meant to explain the porposes of each element in the directory



- Solution: The pipeline that reads data from the training set, trains the model, uses the model to predict the evaluation set, and outputs the predictions on the expected format in a file called "output.csv". All done in this single file running from top to bottom
- Dataset: Includes the development and evalutation dataset.
- Helpers: Includes helper functions such as function for computing mean euclidean distance
- Feature_reduction.ipynb: Has been used to explore the importance of features in the data set in order to figure out which features, if any should be considered dropping
- Find_feature_combination: Used to combare the results of mlp when excluding different combinations of "tmax", "rms" and "area" from the data set
- Grid_hp_search: Does grid search to find the most promising hyperparameters for the chosen model
- Model selection: Has been used to compare the different models that were
- Negpmax_analysis: Used to visualize and explore the outliers of negpmax
- Outlier_detection: Used to explore and detect outliers for different features of the dataset.
- Output: The final predictions from the trained model of "solution.ipynb"
- Visualize_data: used to visualize data, such as the scatter plots included in the report
- Visualize_result uses the trained model on a partioning of the training set to plot the error for each coordinates in a scatter plot. 