
import streamlit as st
import pandas as pd
import io
import os
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.title('Automated Decision Tree Model')

def load_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('File uploaded successfully')
        return df
    else:
        st.write('File Not uploaded')
        return None
    

# Sidebar
def sidebar(df):
    st.sidebar.title("Filter Data")
    # Filter by columns
    st.sidebar.subheader("Filter by Columns")
    selected_columns = st.sidebar.multiselect("Select columns", df.columns)
    if selected_columns:
        df = df[selected_columns]

    return df

def label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data_encoded = df.copy()  # Create a copy of the original data to store the encoded data
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':  # Check if the column contains categorical data
            data_encoded[col] = label_encoder.fit_transform(data_encoded[col])
    return data_encoded   

import pandas as pd


import pandas as pd

def summarize_data(df, columns):
    summary_data = []
    for column in columns:
        unique_values = df[column].astype(str).unique()  # Convert to strings
        value_counts = df[column].value_counts()
        proportion_of_values = df[column].value_counts(normalize=True) * 100
        nan_count = df[column].isna().sum()
        summary_data.append({'Column': column,
                             'Unique values': unique_values,
                             'Value counts': value_counts,
                             'Proportion of values': proportion_of_values,
                             'NaN count': nan_count})
    summary = pd.DataFrame(summary_data)
    return summary

def results( x_test, y_test,x_train, y_train):

    decision_tree1= DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 
    decision_tree= decision_tree1.fit(x_train,y_train)
    plot_tree(decision_tree)
    models = { 'decision tree': decision_tree }
    result = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    for model_name , model in models.items():
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred,average='micro')
        precision = precision_score(y_test, y_pred,average='micro')
        recall = recall_score(y_test, y_pred,average='micro')
        result.loc[model_name] = [type(model).__name__, accuracy, f1, precision, recall]

    return result

def hyperparameter_tuning_decision_tree(x_test, y_test,x_train, y_train ):
    # Create the decision tree classifier
    clf = DecisionTreeClassifier( max_depth=4, min_samples_leaf=10, criterion="gini", random_state=200)
    
    # Train the model
    clf.fit(x_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

##  Main function
def main():
    st.title("Decision Tree")

    st.write("""
    Decision trees are a popular machine learning algorithm used for both classification and regression tasks.
    They work by recursively partitioning the input space into regions, where each region corresponds to a
    specific class label (in classification) or a predicted value (in regression).
    """)

    st.subheader("Entropy")
    st.write("""
    Entropy is a measure of impurity or randomness in a dataset. In the context of decision trees,
    entropy is used to quantify the uncertainty associated with the data at a particular node.
             """)
    st.subheader("Information Gain")
    st.write("""
    Information gain is a measure used to determine the effectiveness of splitting a dataset based on
    a particular feature. It measures the reduction in entropy achieved by splitting the data.
           """)
    # Set title and subtitle
    st.title("DATASET")
    # Load data
    df = load_data()

    if df is not None:
        # Sidebar
        df = sidebar(df)

        # Display data
        st.subheader("Dataset")
        st.write(df)
        # Sidebar with selectbox for choosing information to display
        option = st.sidebar.selectbox("Details of the Dataset", ('Describe', 'Shape', 'Info'))

        # Display selected information
        if st.button('Go and fetch the details of the dataset '):
            if option == 'Describe':
                st.subheader("Dataset Description")
                st.write(df.describe())
            elif option == 'Shape':
                st.subheader("Dataset Shape")
                st.write(df.shape)
            elif option == 'Info':
                st.subheader("Dataset Information")
                with io.StringIO() as buffer:
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    st.text(info_str)

       #         # Perform data preprocessing using LabelEncoder
        encoded_data = label_encoder(df)
        
        # Display the encoded data
        st.subheader("Encoded Data by Label Encoding:")
        if st.button('Encode the given data'):
            st.write(encoded_data)

        st.subheader('VISUALIZATIONS')
        if st.button('visuals'):
            st.subheader('Feature Analysis')
            selected_columns = [col for col in df.columns if col != 'Unique values']
            summary = summarize_data(df, selected_columns)
            st.write(summary)

        st.subheader( 'Group-by')
        target_column = st.selectbox("Select target column", options=df.columns)
             # Select the columns for mean calculation
        selected_columns = st.multiselect("Select columns for mean calculation", options=df.columns)

        if st.button('Calculate Mean Group_By'):
            # Check if any columns are selected
            if not selected_columns:
                st.warning("Please select at least one column for mean calculation.")
            else:
                # Perform group-by operation and calculate mean
                result = df[selected_columns + [target_column]].groupby(target_column).mean()
                st.write(result)

        st.subheader('SUPERVISED LEARNING')
        st.write('DATASET HAVING TARGET/DEPENDENT COLUMN GO FOR SUPERVISED LEARNING')
        # st.sidebar.header('SUPERVISED LEARNING')
        st.subheader('Select Target Column')
        encoded_data = label_encoder(df)
        y = st.selectbox("Select dependent variable", encoded_data.columns)
        st.write(y)
        st.write('If Your dependent Variable is Categorical So Go For Classfication Technique ',  
            'Else Your dependent Variable is Continuous So Go For Regression Technique ')

        st.subheader('TRAIN TEST SPLIT')
        st.sidebar.header('TRAIN TEST SPLIT')
        encoded_data = label_encoder(df)
        X = encoded_data.drop(y, axis = 1)
        Y = encoded_data[y]
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state= 143)
        splits = st.sidebar.selectbox('train test splited datas', ['x_train', 'x_test', 'y_train', 'y_test'])
        if st.button('Show Me'):
            if splits == 'x_train':
                st.write('x_train', x_train)
                st.write('Shape of x_train is', x_train.shape)
            elif splits == 'y_train':
                st.write('y_train', y_train)
                st.write('Shape of y_train is', y_train.shape)
            elif splits == 'x_test':
                st.write('x_test', x_test)
                st.write('Shape of x_test is', x_test.shape)
            elif splits == 'y_test':
                st.write('y_test', y_test)
                st.write('Shape of y_test is', y_test.shape)

        st.subheader('CLASSIFICATION ALGORITHMS')
        if st.button('Model Build'):
            st.write('Classification metrics')
            classification_result = results( x_test, y_test,x_train, y_train)
            st.write(classification_result)
            st.write('Hyper Parameter Tuning')
            Tuning_result = hyperparameter_tuning_decision_tree( x_test, y_test,x_train, y_train)
            st.write(Tuning_result)
            st.write(""" We can see here that by changing the values of the parameter, there is a slight increase in the accuracy.
                      Earlier we were using 'Entropy' as criterion, now we are using 'Gini' as criterion,
                      the max depth of the tree has been increased and so on. Similarly, we can keep on tuning the parameters,
                      to obtain the highest accuracy.
                     """)

if __name__ == "__main__":
    main()