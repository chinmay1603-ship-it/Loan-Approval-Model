import pandas as pd
import os
from sklearn.model_selection import cross_val_score

print(os.getcwd())
print(os.listdir())

data = pd.read_csv('loan_approval_dataset.csv')
data = data.drop(columns = ["loan_id"])
#check for missing values - There are none 
print(data.isnull().sum())

#Finding out which variables are categorical
cat_list = []
for col in data.columns:
    if data[col].dtype == 'object':
        cat_list.append(col)
        

multi_columns = []

# Converting catergorical columns to binary. But if there are more than 2 unique values, then making dummies instead
for column in cat_list:
    if data[column].nunique() == 2:
        values = data[column].unique() 
        data[column] = data[column].map({
            values[0]: 0,
            values[1]: 1
            })
            
    elif data[column].nunique() > 2:
        multi_columns.append(column)

final_data = pd.get_dummies(data,columns=multi_columns, dtype=int)

print(data.columns)        
x = final_data.drop(columns=['loan_status'])
y = final_data['loan_status']
        

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = 0.2, 
    random_state= 4, 
    stratify = y)

#Since we have continous variables, it will be better to use a Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Accuracy of training:", accuracy_score(y_train, y_train_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))


#-------------------------------------------------------Testing with CV
scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring='accuracy'
)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())


#--------------------------------------------------------Checking which features are useful
from sklearn.feature_selection import f_classif

f_scores, p_values = f_classif(x_train, y_train)

results = pd.DataFrame({
    'Feature': x_train.columns,
    'F-score': f_scores,
    'p-value': p_values
})

print(results.sort_values('p-value'))

#Showed that Cibit score and loan term showed very strong relationship with load approval. The others had p_value less than 0.05

#------------------------------------------------------- Testing with different models

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# Model 1: all features
scores_all = cross_val_score(
    GaussianNB(),
    x,
    y,
    cv=5,
    scoring='accuracy'
)

# Model 2: CIBIL score only
scores_cibil = cross_val_score(
    GaussianNB(),
    x[[' cibil_score']],
    y,
    cv=5,
    scoring='accuracy'
)

# Model 3: CIBIL score + loan term
scores_two = cross_val_score(
    GaussianNB(),
    x[[' cibil_score', ' loan_term']],
    y,
    cv=5,
    scoring='accuracy'
)

print("All features:", scores_all.mean())
print("CIBIL only:", scores_cibil.mean())
print("CIBIL + loan term:", scores_two.mean())

#Shows using just 2 of those features gives a higher accuracay compared to using all features. This may be due to other features being very correlated with one another.
#For guassian naive, we assume every feature are conditionally independent

#------------------------------------------------------------------------------- Testing final data:
x = final_data[[' cibil_score', ' loan_term']]
y = final_data['loan_status']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=4,
    stratify=y
)

final_model = GaussianNB()
final_model.fit(x_train, y_train)

y_pred = final_model.predict(x_test)

print("Training accuracy:", model.score(x_train, y_train))
print("Test accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))    

#---------------------------------------- Testing for standard deviation with CV on the final model

scores = cross_val_score(
    final_model,
    x,
    y,
    cv=5,
    scoring='accuracy'
)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

        

        

