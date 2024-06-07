#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[54]:


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score


# # this code takes all files in directory that ends with .csv and merge them into one

# In[2]:


directory_path = "C:/Users/Zahra's computer/Desktop/data mining files"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

print

# Specify the common column based on which you want to merge the files
common_column ='SEQN'

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()
for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)
    print(f"Columns in {csv_file}: {df.columns}")


    # Loop through each CSV file and merge its data into the main DataFrame

file_path = os.path.join(directory_path, csv_file)
merged_data = pd.read_csv(file_path)
    
for csv_file in csv_files[1:]:
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)
    merged_data = pd.merge(merged_data, df, on=common_column,how='outer')

  


# ## Writing data to a CSV file

# In[3]:


path = "C:/Users/Zahra's computer/Desktop/test/merged_data.csv"

merged_data.to_csv(path, index=False)


# ### This code is performing data preprocessing to filter out rows from the DataFrame  based  null values in each row that has more than 50 threshold and check out if it has more than 50% null or not to clear them

# In[4]:


raw_threshold = 50
allcol = len(merged_data.columns)
null_in_col = merged_data.isna().sum(axis=1)
l1 = []
rows_to_keep = null_in_col < (raw_threshold / 100) * allcol
merged_data = merged_data[rows_to_keep]
print(merged_data)


# ### This code is performing data preprocessing to filter out columns from the DataFrame based null values in each column that has more than 50 threshold and check out if it has more than 50% null or not to clear them

# In[5]:


col_threshold = 50
allraw = len(merged_data)
null_in_col = merged_data.isna().sum()
l1 = []
for col, v in null_in_col.items():
    pr = (v / allraw) * 100
    if pr >= col_threshold:
        l1.append(col)

# Drop the selected columns
merged_data = merged_data.drop(columns=l1)

# Display the cleaned DataFrame
print(merged_data)


# ### this is label encoder we have some alphebet in our dataset that must be replaced by numbers

# In[6]:


# Import label encoder 
from sklearn import preprocessing 
df2 = merged_data.select_dtypes(include=['object'])
label_encoder = preprocessing.LabelEncoder() 
merged_data[df2.columns]= label_encoder.fit_transform(X=df2.columns) 
for column in merged_data.columns:
    merged_data=merged_data.fillna(merged_data[column].median())


# #### this code removes columns with outliers and we define threshold=0.5 if they are more than threshold clear them and drop

# In[19]:


def remove_columns_with_outliers(merged_data, threshold=0.3):
    df_cleaned = merged_data.copy()
    for column in merged_data.columns:
        # Calculate the Z-score for each data point in the column
        z_scores = np.abs((merged_data[column] - merged_data[column].mean()) / merged_data[column].std())
        # Identify outliers using the Z-score method
        outliers = z_scores > threshold
        # Remove the column if it has more outliers than the threshold
        if np.sum(outliers) / len(outliers) > threshold:
            df_cleaned = df_cleaned.drop(column, axis=1)
    return df_cleaned
my_dataframe_cleaned = remove_columns_with_outliers(merged_data, threshold=0.05)


# In[20]:


my_dataframe_cleaned 


# ##### normalizing dataset but first of all we should not normalize 'MCQ160L_x', 'MCQ220_x' so we drop themn and then normalize other columns

# In[21]:


columns_to_drop = ['MCQ160L_x', 'MCQ220_x']
merged_data.drop(columns=columns_to_drop)
merged_data
scaler = StandardScaler()
my_dataframe_normalized = pd.DataFrame(scaler.fit_transform(merged_data), columns=merged_data.columns)
print("Normalized DataFrame:")
print(my_dataframe_normalized)

columns_to_drop = ['OHX02CTC', 'OHX03CTC', 'OHX04CTC', 'OHX05CTC', 'OHX06CTC', 'OHX07CTC',
                   'OHX08CTC', 'OHX09CTC', 'OHX10CTC', 'OHX11CTC', 'OHX12CTC', 'OHX13CTC',
                   'OHX14CTC', 'OHX15CTC', 'OHX18CTC', 'OHX19CTC', 'OHX20CTC', 'OHX21CTC',
                   'OHX22CTC', 'OHX23CTC', 'OHX24CTC', 'OHX25CTC', 'OHX26CTC', 'OHX27CTC',
                   'OHX28CTC', 'OHX29CTC', 'OHX30CTC', 'OHX31CTC']

df_dropped = df.drop(columns=columns_to_drop)
print(df_dropped)
# In[24]:


columns_to_drop = ['OHX02CTC', 'OHX03CTC', 'OHX04CTC', 'OHX05CTC', 'OHX06CTC', 'OHX07CTC',
                   'OHX08CTC', 'OHX09CTC', 'OHX10CTC', 'OHX11CTC', 'OHX12CTC', 'OHX13CTC',
                   'OHX14CTC', 'OHX15CTC', 'OHX18CTC', 'OHX19CTC', 'OHX20CTC', 'OHX21CTC',
                   'OHX22CTC', 'OHX23CTC', 'OHX24CTC', 'OHX25CTC', 'OHX26CTC', 'OHX27CTC',
                   'OHX28CTC', 'OHX29CTC', 'OHX30CTC', 'OHX31CTC']

merged_data=merged_data.drop(columns=columns_to_drop)
print(merged_data)


# ##### ploting heat map plot and finding correlation with pearson method and finding top correlated attributes

# In[25]:


threshold=0.8
correlation_matrix = merged_data.corr(method='pearson')
high_correlations=correlation_matrix[(correlation_matrix>threshold)]
# Select the top 100 attributes with the highest correlation above the threshold
top_correlated_attributes = high_correlations.unstack().sort_values(ascending=False).head(100).index
merged_data[top_correlated_attributes.get_level_values(0).unique()].copy()
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

correlation_matrix.to_csv("correlation.csv")


# #### doing feature selection and finding representative of top correlated data

# In[26]:


highly_correlated_pairs = (correlation_matrix.abs() > 0.8) & (correlation_matrix.abs() < 1)
# Choose representatives
representatives = []
for col in merged_data.columns:
    if any(highly_correlated_pairs[col]):
        representatives.append(col)
# Create a new DataFrame with selected representatives
cleaned_df = merged_data[representatives]
new_correlation_matrix = cleaned_df.corr()


# In[27]:


cleaned_df


# ## defining pca for 100 components

# In[28]:


n_components = 100  
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(merged_data)
columns_pca = [f'PC{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(data=X_pca, columns=columns_pca,index=merged_data.index)
print(df_pca)


# In[29]:


merged_data


# ### doing logistic regression for our data set with two columns that are:'MCQ160L_x','MCQ220_x'  Initialize and train the logistic regression model and Make predictions on the test set and evaluate the model

# In[44]:


# Create a new DataFrame with only selected features and the MCQ160L_x 
columns_to_exclude = ['MCQ160L_x']

# Select all features except the specified columns
selected_features = merged_data.drop(columns=columns_to_exclude)
# Print or use the DataFrame 'selected_features'
print(selected_features)
# Extract column names from selected_features and concatenate with 'MCQ160L_x'
data = merged_data[['MCQ160L_x'] + selected_features.columns.tolist()]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[selected_features.columns], data['MCQ160L_x'], test_size=0.2, random_state=42
)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)


# ## evaluation of our regression model and find accuracy of it

# In[59]:


# Create a new DataFrame with only selected features and the target column
columns_to_exclude = ['MCQ160L_x']

# Select all features except the specified columns
selected_features = merged_data.drop(columns=columns_to_exclude)

# Print or use the DataFrame 'selected_features'
print(selected_features)

# Extract column names from selected_features and concatenate with 'MCQ160L_x'
data = merged_data[['MCQ160L_x'] + selected_features.columns.tolist()]

# Drop rows with missing values
data = data.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[selected_features.columns], data['MCQ160L_x'], test_size=0.2, random_state=42
)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model for regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuarcy = accuracy_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"accuarcy: {accuarcy}")


# In[60]:


# Create a new DataFrame with only selected features and the target column
columns_to_exclude = ['MCQ220_x']

# Select all features except the specified columns
selected_features = merged_data.drop(columns=columns_to_exclude)

# Print or use the DataFrame 'selected_features'
print(selected_features)

# Extract column names from selected_features and concatenate with 'MCQ160L_x'
data = merged_data[['MCQ220_x'] + selected_features.columns.tolist()]

# Drop rows with missing values
data = data.dropna()

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    data[selected_features.columns], data['MCQ220_x'], test_size=0.2, random_state=42
)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)



# ### doing LDA dimension reduction for our data set

# In[61]:


# Extract features (X) and target variable (y)
X = merged_data.drop(columns=columns_to_exclude)
y = merged_data['MCQ160L_x']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)


# In[62]:


X = merged_data.drop(columns=columns_to_exclude)
y = merged_data['MCQ220_x']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)


# نتیجه گیری: انتخاب صفات و کاهش ابعاد باعث شده ستون های کمتری داشته باشیم در نتیجه  ارزیابی هم راحتتر خواهد بود اما تاثیرات منفی نیز دارد و باعث از بین رفتن برخی دیتاهای مهم میشود و روش pca دقیقتر ولی سرعت ldaبیشتر است.

# In[ ]:




