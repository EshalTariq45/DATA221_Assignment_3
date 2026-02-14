#question 5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

loaded_dataframe_of_kidney_disease= pd.read_csv("C:/Users/et827/Downloads/kidney_disease.csv")


loaded_dataframe_of_kidney_disease.replace("?", pd.NA, inplace=True) #turning fake missing values into real ones
loaded_dataframe_of_kidney_disease.dropna(inplace=True) #removing incomplete rows


feature_matrix_X=loaded_dataframe_of_kidney_disease.drop("classification", axis=1)

label_vector_y_classification=loaded_dataframe_of_kidney_disease["classification"]

feature_matrix_X= pd.get_dummies(feature_matrix_X, drop_first=True)

feature_matrix_X_train, feature_matrix_X_test, label_vector_y_classification_train, label_vector_y_classification_test= train_test_split(
feature_matrix_X, label_vector_y_classification,
    test_size=0.30,
    random_state=42
)

k_values= [1,3,5,7,9]
accuracy_results=[]

for values in k_values:
    knn_model= KNeighborsClassifier(n_neighbors=values)
    knn_model.fit(feature_matrix_X_train, label_vector_y_classification_train)

    y_prediction= knn_model.predict(feature_matrix_X_test)
    accuracy=accuracy_score(label_vector_y_classification_test, y_prediction)

    accuracy_results.append(accuracy)

print("\nK Value vs Test Accuracy")
print("----------------------------")
for k, accuracy in zip(k_values, accuracy_results):
    print("k=", k, "accuracy=", accuracy)

best_k= k_values[accuracy_results.index(max(accuracy_results))]
print("\nBest k value: ", best_k)
print("Highest Test Accuracy: ", max(accuracy_results))

#changing the value of k affects how flexible or smooth the KNN model is
#   a small k= model looks at very few neighbors, which makes it more sensitive in a sense
#   to individual data points and noise in the training set
#small values of k can cause overfitting because the model will memorize the training
#   data and may not generalize to newer data added
#large values of k will make the model consider a lot of neighbors, which may cause underfitting
#when k is large, models might ignore important patterns and end up classifying most points based on the
#majority class in the dataset