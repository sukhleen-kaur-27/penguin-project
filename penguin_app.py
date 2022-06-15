# Importing the necessary libraries.
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
  pred=model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  species=pred[0]
  if species==0:
    return 'Adelie'
  elif species==1:
    return 'Chinstrap'
  elif species==2:
    return 'Gentoo'
# Design the App
import streamlit as st

st.title('Predicting the Species of Penguin')
bill_length_mm=st.sidebar.slider('Bill Length in mm', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
bill_depth_mm=st.sidebar.slider('Bill Depth in mm', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flipper_length_mm=st.sidebar.slider('Flipper Length in mm', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass_g=st.sidebar.slider('Body Mass in g', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

sex_input=st.sidebar.selectbox('Category Feature Sex Input', ('Male', 'Female'))
if sex_input=='Male':
  sex_input=0
elif sex_input=='Female':
  sex_input=1

island_input=st.sidebar.selectbox('Category Feature Island Input', ('Biscoe', 'Dream', 'Torgersen'))
if island_input=='Biscoe':
  island_input=0
elif island_input=='Dream':
  island_input=1
elif island_input=='Torgersen':
  island_input=2

model=st.sidebar.selectbox('Choose Classifier', ('Support Vector Classifier', 'Logistic Regression', 'Random Forest Classifier'))
if st.sidebar.button('Predict'):
  if model=='Support Vector Classifier':
    species_pred=prediction(svc_model, island_input, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_input)
    score=svc_score
  elif model=='Logistic Regression':
    species_pred=prediction(log_reg, island_input, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_input)
    score=log_reg_score
  elif model=='Random Forest Classifier':
    species_pred=prediction(rf_clf, island_input, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_input)
    score=rf_clf_score

  st.write('Species Predicted is ', species_pred)
  st.write('Accuracy Score is ', score)