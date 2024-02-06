
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph(df):
    # Generate some example data
  

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df[::10], linestyle='-', linewidth=0.5,)
    axs[0].set(xlabel='X-axis', ylabel='Y-axis', title='Lineplot')

    axs[1].hist(df[::10])
    axs[1].set(xlabel='X-axis', ylabel='Y-axis', title='Histogram')

    plt.tight_layout()
    # Display the plot in the Streamlit app
  
    plt.tight_layout()
    # Display the plot in the Streamlit app
    st.pyplot(fig)

def boxplot(df):
     # Create a figure with a single axis
    fig, ax = plt.subplots()

    # Boxplot
    ax.boxplot(df[::10], boxprops=dict(color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), flierprops=dict(markerfacecolor='red', marker='o'))
    ax.set(xlabel='X-axis', ylabel='Y-axis', title='Boxplot')
    plt.xticks(rotation=90)

    # Display the plot in the Streamlit app
    st.pyplot(fig)


def plot_corr(df, initial_var, end_var):
    selected_columns = df.iloc[:, initial_var:end_var]
    # Calculate the correlation matrix
    correlation_matrix = selected_columns.corr()

    # Create a heatmap using seaborn to visualize the correlation matrix
    fig = plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    st.pyplot(fig)