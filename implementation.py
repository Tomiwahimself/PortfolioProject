import streamlit as st
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import pandas as pd
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

def random_forest(df):
    df.index = pd.to_datetime(df['Date'])  
        # drop The original date column
    df = df.drop(['Date'], axis='columns')
        # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
        # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
        # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    split_percentage = 0.8
    split = int(split_percentage*len(df))
  
        # Train data set
    X_train = X[:split]
    y_train = y[:split]
  
        # Test data set
    X_test = X[split:]
    y_test = y[split:]

    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    df['Predicted_Signal'] = rf_gs.predict(X)
    from sklearn import metrics
    #Calculating the accuracy

    test_score = round(rf_gs.score(X_test, y_test), 2)
    train_score = round(rf_gs.score(X_train, y_train), 2)
    accuracy=test_score
    st.write("Accuracy: ", accuracy.round(2))

        # Calculate daily returns
    df['Return'] = df.Close.pct_change()

        # Calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
        # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()

        # Plot Strategy Cumulative returns 
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    plt.style.use('seaborn-darkgrid')
    df=df.dropna() #Dropping the NaN

    chart_data = pd.DataFrame(
    df,
    columns=['Cum_Ret', 'Cum_Strategy'])

    st.line_chart(chart_data)

    
def KNN(df):
    df.index = pd.to_datetime(df['Date'])  
        # drop The original date column
    df = df.drop(['Date'], axis='columns')
        # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
        # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
        # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    split_percentage = 0.8
    split = int(split_percentage*len(df))
  
        # Train data set
    X_train = X[:split]
    y_train = y[:split]
  
        # Test data set
    X_test = X[split:]
    y_test = y[split:]

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    df['Predicted_Signal'] = knn_gs.predict(X)
    from sklearn import metrics
    #Calculating the accuracy

    test_score = round(knn_gs.score(X_test, y_test), 2)
    train_score = round(knn_gs.score(X_train, y_train), 2)
    accuracy=test_score
    st.write("Accuracy: ", accuracy.round(2))

        # Calculate daily returns
    df['Return'] = df.Close.pct_change()

        # Calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
        # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()

        # Plot Strategy Cumulative returns 
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    plt.style.use('seaborn-darkgrid')
    df=df.dropna() # Dropping the NaN

    chart_data = pd.DataFrame(
    df,
    columns=['Cum_Ret', 'Cum_Strategy'])

    st.line_chart(chart_data)



def SVM(df):
    df.index = pd.to_datetime(df['Date'])  
        # drop The original date column
    df = df.drop(['Date'], axis='columns')
        # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
        # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
        # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    split_percentage = 0.8
    split = int(split_percentage*len(df))
  
        # Train data set
    X_train = X[:split]
    y_train = y[:split]
  
        # Test data set
    X_test = X[split:]
    y_test = y[split:]


        # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    df['Predicted_Signal'] = cls.predict(X)
    from sklearn import metrics
    # Calculating the accuracy

    test_score = round(cls.score(X_test, y_test), 2)
    train_score = round(cls.score(X_train, y_train), 2)
    accuracy=test_score
    st.write("Accuracy: ", accuracy.round(2))

        # Calculate daily returns
    df['Return'] = df.Close.pct_change()

        # Calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
        # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()

        # Plot Strategy Cumulative returns 
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    plt.style.use('seaborn-darkgrid')
    df=df.dropna() # Dropping NaN rows
    chart_data = pd.DataFrame(
    df,
    columns=['Cum_Ret', 'Cum_Strategy'])

    st.line_chart(chart_data)# Dsiplaying Line chart


# Title
st.title("Stock Market Dashboard")


#sidebar
sideBar = st.sidebar
display = sideBar.checkbox('Display Dataset')
uploaded_file = sideBar.file_uploader("Upload the dataset")
classifier = sideBar.selectbox('Which Classifier do you want to use?',('SVM' , 'KNN' , 'Random Forest'))

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):

        df=pd.read_csv(uploaded_file)
        if display:
            st.dataframe(df)
        if classifier == 'SVM':
            SVM(df)
        elif classifier == 'Random Forest':
            random_forest(df)
        elif classifier == 'KNN':
            KNN(df)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        if display:
            st.dataframe(df)
        if classifier == 'SVM':
            SVM(df)
        elif classifier == 'Random Forest':
            random_forest(df)
        elif classifier == 'KNN':
            KNN(df)
    else:
        sideBar.write("Please upload csv or excel files only")
