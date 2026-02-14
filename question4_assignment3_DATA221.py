#question 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

#creating KNN classifier k=5
knn=KNeighborsClassifier(n_neighbors=5)

#training the model
knn.fit(feature_matrix_X_train,label_vector_y_classification_train)

#predicting on test data

y_prediction=knn.predict(feature_matrix_X_test)

print("Predictions is: ", y_prediction)

#confusion matrix
confusion_matrix= confusion_matrix(label_vector_y_classification_test, y_prediction)
print("confusion matrix: ", confusion_matrix)

#Accuracy
accuracy= accuracy_score(label_vector_y_classification_test, y_prediction)
print("Accuracy is: ", accuracy)

#Precision
precision= precision_score(label_vector_y_classification_test, y_prediction, pos_label="ckd") #specifying ckd because it will be treated as the positive class
print("Precision is: ", precision)

#Recall
recall= recall_score(label_vector_y_classification_test, y_prediction, pos_label="ckd")
print("Recall: ", recall)

#F1 Score

f1=f1_score(label_vector_y_classification_test, y_prediction, pos_label="ckd")
print("F1-score: ", f1)

#context of kidney disease prediction:
#A true positive means the model has correctly predicted that a patient has chronic kidney disease
#a true negative means the model has correctly predicted that a patient does not have kidney disease
#a false positive means the model predicted kidney disease when the patient doesn't have it
#a false positive means that the model predicted no kidney disease but the patient does have it

#An Accuracy alone may not be enough to evaluate a classification model because it only measures overall
#   correctness. If the data set isn't properly balanced, a model could achieve high accuracy by
#   predicting the majority class while it is still missing important positive cases

#If missing a kidney disease case is very serious, Recall would be the most important metric because it
#   measures how many actual kidney diseases that were identified correctly. Higher recall=reduces the number of
#   false negatives