import streamlit as st
import pandas as pd
import mlflow
from pipeline import *
from utils import *
import streamlit.components.v1 as components
plt.style.use('bmh')

def project_description():
    st.title("Advanced Data Scientist Capstone Project")
    st.subheader("Project Description")
    st.markdown("Creation of an end to end project focused on the creation and deployment of a machine learning model with Streamlit and MLflow.")
    st.image("./images/dataset-cover.jpg")
    st.markdown("---")

def mlflow_ui():
    # MLflow server URL
    mlflow_url = "http://localhost:5000"  # Replace with your MLflow server URL

    # Create the URL for the MLflow UI
    mlflow_ui_url = f"{mlflow_url}/#/"

    # Embed the MLflow UI using an iframe
    components.html(f'<iframe src="{mlflow_ui_url}" width="100%" height="600px"></iframe>', width=900, height=700)

def eda():
    ### Imported data
    train_df = pd.read_csv("./dataset/Test.csv")
    test_df = pd.read_csv("./dataset/Train.csv")

    ### EDA
    st.title("EDA")
    st.header("Initial Visualizations")
    st.markdown("The decision to study the following dataset is due to comes from an e ngieniering source with purpose to develop methodologies to help identify failures in the area of predictive maintenance. ")
    st.markdown("The dataset has around 41 variables including the target variable. There is not much information about the features in the repository so, we will have a look to some visualizations to properly understand what we have. ")
    st.markdown("All the features are numerical according to the type of the variables. The only categorical is the target variable that is presented as binary feature made from 0 and 1. Threfore there is no need to transform any variable into numerical.")
    st.write("Kaggle Repository [Link to Kaggle Repo](https://www.kaggle.com/datasets/mariyamalshatta/renewind?resource=download)")
    st.dataframe(train_df.head())

    st.subheader("Lineplot and Histogram")
    st.markdown("The histogram indicates visually that the features a normal distribution pattern.")
    plot_graph(train_df)
    st.markdown("---")

    ### Boxplot
    st.subheader("Boxplot")
    st.markdown("The following boxplot presents basic staistics of how the signals are distributed and the red dots are the outliers presented in each variable")
    boxplot(train_df)
    st.markdown("---")

    ### Corr matrix
    st.subheader("Correlation Matrix")
    st.markdown("The high dimensionality of the dataset makes somehow a difficulty to study in depth the correlation matrix. There is a possibility to play around with algorithms that can reduce the domensionality of the datasets, but let's play with the graph for now. ")

    col1, col2 = st.columns(2)
    with col1:
        input1 = st.number_input("Enter an initial value:", min_value=0, max_value=41, value=0)

    with col2:
        input2 = st.number_input("Enter an end-value:", min_value=0, max_value=41, value=10)
        

    if st.button('Generate the correlation matrix.'):
        plot_corr(train_df, initial_var=input1, end_var=input2)

    st.markdown("---")

def model_deployment():
    st.header("Model Deployment")
    st.markdown("The following section will create the model and store it inside of the app. This algorithms are only trained if there are new data to feed into the algorithm, with the intention to automatize the whole process of reaching the most accurate model.")
    
    # Add MLflow integration
    with st.expander("MLflow Integration"):
       
        selected_run_id = st.selectbox("Select a run from MLflow", mlflow.search_runs()["run_id"])

        # Display details of the selected run
        if selected_run_id:
            selected_run = mlflow.get_run(selected_run_id)
            st.write("Selected Run Details:")
            st.write(f"Run ID: {selected_run.info.run_id}")
            st.write(f"Start Time: {selected_run.info.start_time}")
            st.write(f"End Time: {selected_run.info.end_time}")

            # Display metrics and parameters for the selected run
            st.write("Metrics:")
            metrics = selected_run.data.metrics
            st.write(metrics)

            st.write("Parameters:")
            params = selected_run.data.params
            st.write(params)

    # Add MLflow UI section
    
    
def main():
    
    selected_option = st.sidebar.selectbox('Navigate to:', ['Project Description', 'EDA', 'Model Deployment', "About Me"])
     # Display content based on the selected option
    if selected_option == 'Project Description':
        project_description()
    elif selected_option == 'EDA':
        eda()
    elif selected_option == 'Model Deployment':
        st.subheader("Run Machine Learning Pipeline")
        st.write("Enter the experiment name:")
        experiment_name = st.text_input("Experiment Name", "Your_Default_Experiment_Name")

        if st.button("Run Machine Learning Pipeline"):
            test_csv = "./dataset/Test.csv"
            train_csv = "./dataset/Train.csv"
            
            if experiment_name:
                # set the experiment id
                mlflow.set_experiment(experiment_id="0")
                creation_pipeline(test_csv, train_csv, experiment_name)
                st.success("Machine Learning Pipeline has been executed!")

            model_deployment()
            st.subheader("MLflow UI")
            mlflow_ui()
        
 
if __name__ == '__main__':
    main()