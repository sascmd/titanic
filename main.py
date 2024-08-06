import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Titanic veri setini yükle
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Basit veri ön işleme
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data.dropna()

# Özellikler ve hedef değişken
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model ile tahmin yap
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare):
    features = [[Pclass, Sex, Age, SibSp, Parch, Fare]]
    prediction = model.predict(features)
    return prediction[0]

# Streamlit uygulaması
st.title('Titanic Survival Prediction')

st.sidebar.header('User Input')

Pclass = st.sidebar.slider('Pclass', min_value=1, max_value=3, value=1)
Sex = st.sidebar.radio('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
Age = st.sidebar.number_input('Age', min_value=0, value=22, step=1)
SibSp = st.sidebar.slider('SibSp', min_value=0, max_value=10, value=0)
Parch = st.sidebar.slider('Parch', min_value=0, max_value=10, value=0)
Fare = st.sidebar.number_input('Fare', min_value=0.0, value=7.25, step=0.1)

if st.sidebar.button('Predict'):
    survival = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare)
    st.write('Survived' if survival == 1 else 'Not Survived')
