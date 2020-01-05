import streamlit as st


# IMPORTING DATA
import pandas as pd
from pandas import DataFrame
import numpy as np

# IMPORTS FOR VISUALIZATIONS
import matplotlib.pyplot as plt
import seaborn as sns


st.title('BREAST CANCER CLASSIFICATION MODEL')
from PIL import Image

img = Image.open('Breast_cancer.jpg ')
st.image(img, width = 300)

# Todays Date
import datetime
st.date_input('Today date', datetime.datetime.now())

# Text Input
Fullname = st.text_input('Enter Your Fullname, Surname First', 'Type Here')
if st.button("Summit"):
    result = 'Hello! ' + Fullname.title()
    st.success(result)


# Lets Load Our Data
from sklearn.datasets import load_breast_cancer
cancer =  load_breast_cancer()

# Fxn to Load Dataset
@st.cache
def load_data(cancer):
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))
    return df

cancer = load_data(cancer) 

if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.dataframe(cancer.head())
    if st.button("Tail"):
        st.dataframe(cancer.tail())

# All Dataset
if st.checkbox("Entire Dataset"):
    st.dataframe(cancer)
# Desc
if st.checkbox("Description"):
    st.dataframe(cancer.describe())


st.subheader("Let's make a plot to see how few features are correlated")
if st.checkbox('Plot of column headers'):
    st.write(sns.pairplot(cancer,hue = 'target', vars = ['mean radius', 'mean texture','mean area','mean perimeter', 'mean smoothness']))
    st.pyplot()
    st.write('from the above plot, we can see that 0 indicates Milignant and 1 indicates Benign')


# Model Training Import
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



X =  cancer.drop(['target'], axis = 1)
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 5)

# Lets scale our train_data -- Normalization 
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

# Lets scale our text_data -- Normalization 
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_train)/range_test

st.subheader("Let's make a plot to display the number of Malignant Vs Benign cases")
if st.checkbox('Malignant Vs Benign cases'):
    st.write(sns.countplot(y_train))
    st.pyplot()
    st.write('from the above plot, we can see that Benign cases are much higher than Malignant cases')


st.subheader("Let's make a plot to see our  the correlation between MEAN RADIUS and MEAN SMOOTHNESS")
if st.checkbox('Plot of MEAN RADIUS Vs MEAN SMOOTHNESS'):
    st.write(sns.scatterplot(X_train_scaled['mean radius'], X_train_scaled['mean smoothness'], hue = y_train))
    st.pyplot()
   

st.subheader("Let's make a plot to see our features are correlated")
if st.checkbox('Plot of Feature correlations'):
    plt.figure(figsize=(20,10))
    st.write(sns.heatmap(X_train_scaled.corr(),annot = True))
    st.pyplot()
    st.write('')

# IMPROVING THE MODEL

param_grid = {'C': [0.1, 1, 10,100], 'gamma':[1, 0.1, 0.01, 0.001], 
                     'kernel' : ['rbf'] }

# Instance of our model
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)

grid.fit(X_train_scaled, y_train)

y_predict = grid.predict(X_test_scaled)
  

# LET MAKE A CONFUSION MATRIX

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)


class_names = ['Manignant','Benign']

st.subheader("Let's make a confusion matrix to tell how good our predictions are... ")
if st.checkbox('Confusion matrix plot of our cancer model'):
    # Plot non-normalized confusion matrix
    if st.button('Confusion matrix, without normalization'):
        plot_confusion_matrix(y_test, y_predict.round(), classes=class_names,title='Confusion matrix, without normalization')
        st.pyplot()
       
    
    # Plot normalized confusion matrix
    if st.button('Normalized confusion matrix'):
        plot_confusion_matrix(y_test, y_predict.round(), classes=class_names, normalize=True,title='Normalized confusion matrix')
        st.pyplot()


class_report = classification_report(y_predict,y_test)
Accuracy = accuracy_score(y_test,y_predict)


if st.checkbox("Lets see the details of our model Prediction"):
    if st.button("Text Prediction"):
        st.write(class_report)
        st.write('Accuracy of our Model is : ', Accuracy)

  

# Lets make a new predictions with a new dataset
@st.cache()
def Manignant_or_Benign(X):
    if (grid.predict([X])) == 0:
        print('You\'re looking at Manignant !')
    else:
        print('You\'re looking at Benign !')

m = [17.9900,10.3800,122.8000,1001,0.1184,0.2776, 
         0.3001,0.1471,0.2419,0.0787,1.0950, 0.9053, 8.5890 ,
         153.4000, 0.0064, 0.0490,0.0537,0.0159,0.0300,0.0062,25.3800,
         17.3300,184.6000, 2019, 0.1622, 0.6656, 0.7119,0.2654,0.4601, 0.1189 ]

if st.checkbox("Lets Make New Predictions"):
    if st.button("New Predictions"):
        st.write(Manignant_or_Benign(m))

# Ballon
st.balloons()

# About
st.sidebar.subheader("About App")
st.sidebar.text("Breast Cancer Classification App with Streamlit")
st.sidebar.info("Thanks to the Streamlit Team")


st.sidebar.subheader("By")
st.sidebar.text("Okolie Chukwuka Albert")
st.sidebar.text("Portfolio link : https://chukypedro.netlify.com")




        
