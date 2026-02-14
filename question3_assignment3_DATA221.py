#question 3
import pandas as pd
from sklearn.model_selection import train_test_split

loaded_dataframe_of_kidney_disease= pd.read_csv("C:/Users/et827/Downloads/kidney_disease.csv")

feature_matrix_X=loaded_dataframe_of_kidney_disease.drop("classification", axis=1)

label_vector_y_classification=loaded_dataframe_of_kidney_disease["classification"]

feature_matrix_X_train, feature_matrix_X_test, label_vector_y_classification_train, label_vector_y_classification_test= train_test_split(
feature_matrix_X, label_vector_y_classification,
    test_size=0.30,
    random_state=42
)

#we should not train and test a model on the same data because the model would just memorize
#   the training data instead. We want the model to learn general patterns, so training and testing the model
#   on the same data would not be a good outcome if we want to reflect a real world performance

#the purpose of the testing set is to determine how well the model generalizes to new and unseen data.
#   it would provide an unbiased estimate on the models performance and determines if the model is overfitting
#   or not