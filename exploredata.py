import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Tuple, Optional, Union, Any
from openai import OpenAI

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning & EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6eef1;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenRouter client
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )
except KeyError:
    st.error("OpenRouter API key not found. Please configure it in Streamlit secrets.")
    st.stop()

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'data_insights' not in st.session_state:
    st.session_state.data_insights = None

# Sidebar configuration
with st.sidebar:
    st.title("MCP Data Server")
    
    # Data upload section
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Load sample data option
    if st.button("Load Sample Data"):
        data = {
            'Age': [25, 30, 35, 40, np.nan, 45, 50, 55, 60, 65],
            'Income': [50000, 60000, np.nan, 80000, 90000, 100000, np.nan, 120000, 130000, 140000],
            'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', np.nan],
            'Education': ['Bachelor', 'Master', 'PhD', np.nan, 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
            'Satisfaction': [7, 8, 9, 6, np.nan, 8, 7, 9, np.nan, 8]
        }
        st.session_state.df = pd.DataFrame(data)
        st.session_state.original_df = st.session_state.df.copy()
        st.session_state.cleaning_history = []
        st.success("Sample data loaded!")
    
    # Reset data button
    if st.session_state.df is not None:
        if st.button("Reset Data"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.cleaning_history = []
            st.success("Data reset to original state!")

# Process uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        st.session_state.original_df = st.session_state.df.copy()
        st.session_state.cleaning_history = []
        st.success(f"File uploaded successfully: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Define helper functions for OpenRouter API
def query_openrouter(prompt: str) -> str:
    """Send a prompt to OpenRouter API and return the response"""
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_data_insights(df: pd.DataFrame) -> str:
    """Get insights about the data using OpenRouter"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    desc_stats = df.describe(include='all').to_string()
    
    try:
        corr_str = df.select_dtypes(include=[np.number]).corr().to_string()
    except:
        corr_str = "Could not compute correlations."
    
    prompt = f"""
    I have a dataset with the following characteristics:
    
    Data Info:
    {info_str}
    
    Descriptive Statistics:
    {desc_stats}
    
    Correlation Matrix:
    {corr_str}
    
    Please provide insights about this dataset covering:
    1. Basic structure and quality assessment
    2. Key statistics and distributions
    3. Notable correlations or patterns
    4. Potential issues like missing values, outliers, or imbalances
    5. Suggestions for data cleaning and preprocessing
    
    Keep your analysis concise and focused on the most important aspects.
    """
    
    with st.spinner("Generating data insights with LLM..."):
        insights = query_openrouter(prompt)
    
    return insights

def suggest_cleaning_steps(df: pd.DataFrame) -> str:
    """Suggest data cleaning steps using OpenRouter"""
    missing_values = df.isnull().sum().to_string()
    duplicates = df.duplicated().sum()
    
    prompt = f"""
    I need to clean a dataset with the following characteristics:
    
    Missing Values by Column:
    {missing_values}
    
    Number of Duplicate Rows: {duplicates}
    
    Column Data Types:
    {df.dtypes.to_string()}
    
    Provide a step-by-step data cleaning plan addressing:
    1. How to handle missing values for each column
    2. How to deal with duplicates if present
    3. Data type conversions if needed
    4. Any other preprocessing steps you recommend
    
    Use a step-by-step format with clear, actionable instructions.
    """
    
    with st.spinner("Generating cleaning suggestions with LLM..."):
        suggestions = query_openrouter(prompt)
    
    return suggestions

def get_data_explanation(df: pd.DataFrame) -> str:
    """Get an explanation of what the data represents using OpenRouter"""
    sample = df.head(5).to_string()
    
    prompt = f"""
    Based on this sample of data:
    
    {sample}
    
    And these column names: {', '.join(df.columns)}
    
    Please explain what kind of dataset this might be, what each column likely represents, 
    and what kind of analysis could be performed with it.
    """
    
    with st.spinner("Generating data explanation with LLM..."):
        explanation = query_openrouter(prompt)
    
    return explanation

def recommend_visualizations(df: pd.DataFrame) -> str:
    """Recommend visualizations for the data using OpenRouter"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    prompt = f"""
    I have a dataset with the following column types:
    
    Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
    Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
    
    Please recommend 5-7 specific visualizations that would be most informative for exploring this data,
    explaining why each visualization is useful and what insights it might reveal.
    Format each recommendation as:
    1. [Visualization Type]: [Brief explanation]
    """
    
    with st.spinner("Generating visualization recommendations with LLM..."):
        recommendations = query_openrouter(prompt)
    
    return recommendations

# Main app layout
st.title("MCP Data Cleaning & EDA with LLM Assistant")

# Check if data is loaded
if st.session_state.df is not None:
    # Create tabs for different functionalities
    tabs = st.tabs(["Data Overview", "Data Cleaning", "EDA & Visualization", "LLM Assistant"])
    
    # Data Overview Tab
    with tabs[0]:
        st.header("Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            st.subheader("Data Information")
            buffer = io.StringIO()
            st.session_state.df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.subheader("Dataset Statistics")
            st.write(f"**Rows:** {st.session_state.df.shape[0]}")
            st.write(f"**Columns:** {st.session_state.df.shape[1]}")
            st.write(f"**Missing Values:** {st.session_state.df.isnull().sum().sum()}")
            st.write(f"**Duplicate Rows:** {st.session_state.df.duplicated().sum()}")
            
            st.subheader("Column Data Types")
            dtype_df = pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type'])
            dtype_df.index.name = 'Column'
            st.dataframe(dtype_df, use_container_width=True)
            
            if st.button("Get Data Explanation"):
                explanation = get_data_explanation(st.session_state.df)
                st.info(explanation)
    
    # Data Cleaning Tab
    with tabs[1]:
        st.header("Data Cleaning")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Data preview
            st.subheader("Current Data")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            # Cleaning operations
            st.subheader("Cleaning Operations")
            
            cleaning_tab1, cleaning_tab2 = st.tabs(["Basic Cleaning", "Advanced Cleaning"])
            
            with cleaning_tab1:
                # Handle missing values
                st.subheader("Handle Missing Values")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    missing_col = st.selectbox("Select Column", st.session_state.df.columns)
                
                with col_b:
                    missing_action = st.selectbox("Action", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"])
                
                fill_value = None
                if missing_action == "Fill with value":
                    fill_value = st.text_input("Value to fill")
                
                if st.button("Apply Missing Value Treatment"):
                    try:
                        df_copy = st.session_state.df.copy()
                        
                        if missing_action == "Drop rows":
                            st.session_state.df = st.session_state.df.dropna(subset=[missing_col])
                            action_desc = f"Dropped rows with missing values in '{missing_col}'"
                        
                        elif missing_action == "Fill with mean":
                            if pd.api.types.is_numeric_dtype(st.session_state.df[missing_col]):
                                st.session_state.df[missing_col] = st.session_state.df[missing_col].fillna(st.session_state.df[missing_col].mean())
                                action_desc = f"Filled missing values in '{missing_col}' with mean"
                            else:
                                st.error("Cannot fill with mean: column is not numeric")
                                action_desc = None
                        
                        elif missing_action == "Fill with median":
                            if pd.api.types.is_numeric_dtype(st.session_state.df[missing_col]):
                                st.session_state.df[missing_col] = st.session_state.df[missing_col].fillna(st.session_state.df[missing_col].median())
                                action_desc = f"Filled missing values in '{missing_col}' with median"
                            else:
                                st.error("Cannot fill with median: column is not numeric")
                                action_desc = None
                        
                        elif missing_action == "Fill with mode":
                            st.session_state.df[missing_col] = st.session_state.df[missing_col].fillna(st.session_state.df[missing_col].mode()[0])
                            action_desc = f"Filled missing values in '{missing_col}' with mode"
                        
                        elif missing_action == "Fill with value":
                            try:
                                if pd.api.types.is_numeric_dtype(st.session_state.df[missing_col]):
                                    fill_value = float(fill_value)
                                st.session_state.df[missing_col] = st.session_state.df[missing_col].fillna(fill_value)
                                action_desc = f"Filled missing values in '{missing_col}' with '{fill_value}'"
                            except ValueError:
                                st.error("Invalid fill value for the column type")
                                action_desc = None
                        
                        if action_desc:
                            st.session_state.cleaning_history.append({
                                'action': action_desc,
                                'previous_state': df_copy
                            })
                            st.success(action_desc)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                # Handle duplicates
                st.subheader("Handle Duplicates")
                if st.button("Remove Duplicate Rows"):
                    df_copy = st.session_state.df.copy()
                    initial_count = len(st.session_state.df)
                    st.session_state.df = st.session_state.df.drop_duplicates()
                    final_count = len(st.session_state.df)
                    removed = initial_count - final_count
                    
                    if removed > 0:
                        action_desc = f"Removed {removed} duplicate rows"
                        st.session_state.cleaning_history.append({
                            'action': action_desc,
                            'previous_state': df_copy
                        })
                        st.success(action_desc)
                    else:
                        st.info("No duplicate rows found")
            
            with cleaning_tab2:
                # Data type conversion
                st.subheader("Convert Data Types")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    convert_col = st.selectbox("Select Column for Conversion", st.session_state.df.columns)
                
                with col_b:
                    convert_type = st.selectbox("New Type", ["int", "float", "string", "category", "datetime"])
                
                if st.button("Convert Data Type"):
                    try:
                        df_copy = st.session_state.df.copy()
                        
                        if convert_type == "int":
                            st.session_state.df[convert_col] = pd.to_numeric(st.session_state.df[convert_col], errors='coerce').astype('Int64')
                        elif convert_type == "float":
                            st.session_state.df[convert_col] = pd.to_numeric(st.session_state.df[convert_col], errors='coerce')
                        elif convert_type == "string":
                            st.session_state.df[convert_col] = st.session_state.df[convert_col].astype(str)
                        elif convert_type == "category":
                            st.session_state.df[convert_col] = st.session_state.df[convert_col].astype('category')
                        elif convert_type == "datetime":
                            st.session_state.df[convert_col] = pd.to_datetime(st.session_state.df[convert_col], errors='coerce')
                        
                        action_desc = f"Converted '{convert_col}' to {convert_type}"
                        st.session_state.cleaning_history.append({
                            'action': action_desc,
                            'previous_state': df_copy
                        })
                        st.success(action_desc)
                    except Exception as e:
                        st.error(f"Conversion error: {str(e)}")
                
                # Create new column
                st.subheader("Create New Column")
                new_col_name = st.text_input("New Column Name")
                formula = st.text_input("Formula (e.g., Age * 2 or Income + 1000)", help="Use column names without 'df[' prefix")
                
                if st.button("Create Column"):
                    try:
                        if new_col_name and formula:
                            df_copy = st.session_state.df.copy()
                            st.session_state.df[new_col_name] = st.session_state.df.eval(formula, engine='python')
                            action_desc = f"Created new column '{new_col_name}' with formula: {formula}"
                            st.session_state.cleaning_history.append({
                                'action': action_desc,
                                'previous_state': df_copy
                            })
                            st.success(action_desc)
                        else:
                            st.warning("Please enter both column name and formula")
                    except Exception as e:
                        st.error(f"Error creating column: {str(e)}")
                
                # Filter rows
                st.subheader("Filter Rows")
                filter_condition = st.text_input("Filter condition (e.g., Age > 30)", help="Use column names without 'df[' prefix")
                
                if st.button("Apply Filter"):
                    try:
                        if filter_condition:
                            df_copy = st.session_state.df.copy()
                            filtered_df = st.session_state.df.query(filter_condition, engine='python')
                            if len(filtered_df) > 0:
                                st.session_state.df = filtered_df
                                action_desc = f"Filtered rows using condition: {filter_condition}"
                                st.session_state.cleaning_history.append({
                                    'action': action_desc,
                                    'previous_state': df_copy
                                })
                                st.success(f"Applied filter. Rows remaining: {len(st.session_state.df)}")
                            else:
                                st.error("Filter resulted in empty DataFrame. No changes applied.")
                        else:
                            st.warning("Please enter a filter condition")
                    except Exception as e:
                        st.error(f"Error applying filter: {str(e)}")
        
        with col2:
            # LLM cleaning suggestions
            st.subheader("LLM Cleaning Suggestions")
            if st.button("Get Cleaning Suggestions"):
                suggestions = suggest_cleaning_steps(st.session_state.df)
                st.info(suggestions)
            
            # Cleaning history
            st.subheader("Cleaning History")
            if st.session_state.cleaning_history:
                for i, action in enumerate(reversed(st.session_state.cleaning_history)):
                    with st.expander(f"{len(st.session_state.cleaning_history) - i}. {action['action']}"):
                        if st.button("Undo this action", key=f"undo_{i}"):
                            st.session_state.df = action['previous_state']
                            st.session_state.cleaning_history = st.session_state.cleaning_history[:-i-1]
                            st.success("Action undone!")
                            st.experimental_rerun()
            else:
                st.info("No cleaning actions performed yet")
            
            # Export cleaned data
            st.subheader("Export Cleaned Data")
            export_format = st.selectbox("Export Format", ["CSV", "Excel"])
            
            if st.button("Export Data"):
                try:
                    if export_format == "CSV":
                        csv = st.session_state.df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif export_format == "Excel":
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            st.session_state.df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                        b64 = base64.b64encode(output.getvalue()).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cleaned_data.xlsx">Download Excel File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
    
    # EDA & Visualization Tab
    with tabs[2]:
        st.header("Exploratory Data Analysis & Visualization")
        
        viz_tab1, viz_tab2 = st.tabs(["Auto EDA", "Custom Visualization"])
        
        with viz_tab1:
            st.subheader("Automated EDA")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("Generate Data Insights"):
                    with st.spinner("Analyzing data..."):
                        st.session_state.data_insights = get_data_insights(st.session_state.df)
                
                if st.button("Recommend Visualizations"):
                    with st.spinner("Generating recommendations..."):
                        viz_recommendations = recommend_visualizations(st.session_state.df)
                        st.session_state.viz_recommendations = viz_recommendations
            
            with col2:
                if st.session_state.data_insights:
                    st.subheader("Data Insights")
                    st.markdown(st.session_state.data_insights)
                
                if 'viz_recommendations' in st.session_state:
                    st.subheader("Recommended Visualizations")
                    st.markdown(st.session_state.viz_recommendations)
            
            st.subheader("Standard Plots")
            
            st.write("### Missing Values Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(st.session_state.df.isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.write("### Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = st.session_state.df[numeric_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
            
            if len(numeric_cols) > 0:
                st.write("### Distribution of Numeric Variables")
                for col in numeric_cols[:5]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(st.session_state.df[col].dropna(), kde=True, ax=ax)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.write("### Distribution of Categorical Variables")
                for col in cat_cols[:5]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = st.session_state.df[col].value_counts()
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                        plt.title(f'Top 10 Categories in {col}')
                    else:
                        plt.title(f'Categories in {col}')
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with viz_tab2:
            st.subheader("Custom Visualization")
            
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Violin Plot", "Pair Plot", "Heat Map"]
            )
            
            if plot_type in ["Scatter Plot", "Line Plot"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", st.session_state.df.columns)
                with col2:
                    y_col = st.selectbox("Y-axis", st.session_state.df.columns)
                with col3:
                    hue_col = st.selectbox("Color by (optional)", ["None"] + list(st.session_state.df.columns))
                
                if st.button("Generate Plot"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if hue_col != "None":
                        if plot_type == "Scatter Plot":
                            sns.scatterplot(data=st.session_state.df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                        else:
                            sns.lineplot(data=st.session_state.df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                    else:
                        if plot_type == "Scatter Plot":
                            sns.scatterplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                        else:
                            sns.lineplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                    
                    plt.title(f"{plot_type} of {y_col} vs {x_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif plot_type in ["Bar Plot"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", st.session_state.df.columns)
                with col2:
                    y_col = st.selectbox("Y-axis (count or value)", ["count"] + list(st.session_state.df.columns))
                with col3:
                    hue_col = st.selectbox("Group by (optional)", ["None"] + list(st.session_state.df.columns))
                
                if st.button("Generate Bar Plot"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if y_col == "count":
                        if hue_col != "None":
                            sns.countplot(data=st.session_state.df, x=x_col, hue=hue_col, ax=ax)
                        else:
                            sns.countplot(data=st.session_state.df, x=x_col, ax=ax)
                        plt.title(f"Count of {x_col}")
                    else:
                        if hue_col != "None":
                            sns.barplot(data=st.session_state.df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                        else:
                            sns.barplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                        plt.title(f"{y_col} by {x_col}")
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif plot_type == "Histogram":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Column", st.session_state.df.select_dtypes(include=[np.number]).columns)
                with col2:
                    bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
                
                if st.button("Generate Histogram"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=st.session_state.df, x=x_col, bins=bins, kde=True, ax=ax)
                    plt.title(f"Histogram of {x_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif plot_type in ["Box Plot", "Violin Plot"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("Category (X-axis)", ["None"] + list(st.session_state.df.select_dtypes(include=['object', 'category']).columns))
                with col2:
                    y_col = st.selectbox("Value (Y-axis)", st.session_state.df.select_dtypes(include=[np.number]).columns)
                with col3:
                    hue_col = st.selectbox("Split by (optional)", ["None"] + list(st.session_state.df.columns))
                
                if st.button(f"Generate {plot_type}"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if x_col == "None":
                        if plot_type == "Box Plot":
                            sns.boxplot(data=st.session_state.df, y=y_col, ax=ax)
                        else:
                            sns.violinplot(data=st.session_state.df, y=y_col, ax=ax)
                    else:
                        if hue_col != "None":
                            if plot_type == "Box Plot":
                                sns.boxplot(data=st.session_state.df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                            else:
                                sns.violinplot(data=st.session_state.df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                        else:
                            if plot_type == "Box Plot":
                                sns.boxplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                            else:
                                sns.violinplot(data=st.session_state.df, x=x_col, y=y_col, ax=ax)
                    
                    plt.title(f"{plot_type} of {y_col}" + (f" by {x_col}" if x_col != "None" else ""))
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif plot_type == "Pair Plot":
                col1, col2 = st.columns(2)
                with col1:
                    num_cols = st.multiselect(
                        "Select columns (numeric)",
                        options=st.session_state.df.select_dtypes(include=[np.number]).columns,
                        default=list(st.session_state.df.select_dtypes(include=[np.number]).columns)[:3]
                    )
                with col2:
                    hue_col = st.selectbox("Color by", ["None"] + list(st.session_state.df.columns))
                
                if st.button("Generate Pair Plot"):
                    if len(num_cols) < 2:
                        st.error("Please select at least 2 columns for the pair plot")
                    else:
                        with st.spinner("Generating Pair Plot..."):
                            if hue_col != "None":
                                pair_plot = sns.pairplot(st.session_state.df[num_cols + [hue_col]], hue=hue_col)
                            else:
                                pair_plot = sns.pairplot(st.session_state.df[num_cols])
                            st.pyplot(pair_plot.fig)
            
            elif plot_type == "Heat Map":
                corr_method = st.selectbox("Correlation Method", ["pearson", "kendall", "spearman"])
                
                if st.button("Generate Heat Map"):
                    numeric_df = st.session_state.df.select_dtypes(include=[np.number])
                    
                    if numeric_df.shape[1] < 2:
                        st.error("Not enough numeric columns for correlation analysis")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        corr = numeric_df.corr(method=corr_method)
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
                        plt.title(f"Correlation Heatmap ({corr_method})")
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # LLM Assistant Tab
    with tabs[3]:
        st.header("LLM Assistant")
        
        st.write("Ask the LLM anything about your data or get custom analysis.")
        
        # User input for custom prompt
        user_prompt = st.text_area("Enter your question or prompt:", 
                                 help="Examples: 'What trends can you identify?' or 'Suggest a machine learning approach for this data.'")
        
        if st.button("Submit to LLM"):
            if user_prompt.strip():
                with st.spinner("Waiting for LLM response..."):
                    # Include data summary in the prompt for context
                    sample = st.session_state.df.head(5).to_string()
                    full_prompt = f"Dataset sample:\n{sample}\n\nColumn names: {', '.join(st.session_state.df.columns)}\n\nUser question: {user_prompt}"
                    response = query_openrouter(full_prompt)
                    st.markdown("### LLM Response")
                    st.markdown(response)
            else:
                st.warning("Please enter a prompt.")
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Summarize Data"):
                with st.spinner("Generating summary..."):
                    prompt = "Provide a concise summary of the dataset based on the sample and column names."
                    sample = st.session_state.df.head(5).to_string()
                    full_prompt = f"Dataset sample:\n{sample}\n\nColumn names: {', '.join(st.session_state.df.columns)}\n\n{prompt}"
                    response = query_openrouter(full_prompt)
                    st.info(response)
        
        with col2:
            if st.button("Suggest Analysis"):
                with st.spinner("Generating suggestions..."):
                    prompt = "Suggest potential analyses or statistical tests that could be performed on this dataset."
                    sample = st.session_state.df.head(5).to_string()
                    full_prompt = f"Dataset sample:\n{sample}\n\nColumn names: {', '.join(st.session_state.df.columns)}\n\n{prompt}"
                    response = query_openrouter(full_prompt)
                    st.info(response)

else:
    st.info("Please upload a dataset or load sample data to begin.")
