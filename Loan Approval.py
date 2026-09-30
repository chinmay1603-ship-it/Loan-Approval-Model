import pandas as pd
import os

print(os.getcwd())
print(os.listdir())

data = pd.read_csv('loan_approval_dataset.csv')

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

        

        

