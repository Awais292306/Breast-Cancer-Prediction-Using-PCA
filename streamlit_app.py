import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer PCA & Logistic Regression Analysis",
    page_icon="📊",
    layout="wide"
)

def load_and_preprocess_data():
    """Load the breast cancer dataset and preprocess it"""
    # Load the dataset
    df = pd.read_csv('Breast_cancer_dataset.csv')
    
    # Drop unnecessary columns
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    
    # Convert diagnosis to binary (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    return X, y, df

def perform_pca_analysis(X, n_components=None):
    """Perform PCA analysis on the dataset"""
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    if n_components is None:
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate cumulative explained variance ratio
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        
        # Use either 2 components for visualization or optimal for 95% variance
        n_comp_final = min(n_components_95, 2)
        pca_final = PCA(n_components=n_comp_final)
        X_pca_final = pca_final.fit_transform(X_scaled)
        
        return X_pca_final, pca_final, scaler, X_scaled, n_components_95
    else:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, pca, scaler, X_scaled, None

def logistic_regression_analysis(X_train, X_test, y_train, y_test):
    """Perform logistic regression analysis"""
    # Train logistic regression model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return lr_model, y_pred, accuracy

def create_visualizations(df, X_pca, y, pca_model, y_test, y_pred):
    """Create various visualizations for the analysis"""
    figs = {}
    
    # PCA Visualization
    if X_pca.shape[1] >= 2:
        fig_pca = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            color=y,
            labels={'x': f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)', 
                    'y': f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)'},
            title='PCA: First Two Principal Components',
            color_discrete_sequence=['red', 'blue'],
            category_orders={0: [0, 1]},
            hover_data={'Diagnosis': [f'Malignant' if val == 1 else 'Benign' for val in y]}
        )
        fig_pca.update_layout(width=700, height=500)
        figs['pca'] = fig_pca
    
    # Explained Variance Plot
    explained_variance = pca_model.explained_variance_ratio_
    fig_variance = go.Figure(data=[
        go.Bar(x=[f'PC{i+1}' for i in range(len(explained_variance))], 
               y=explained_variance,
               name='Individual Explained Variance')
    ])
    fig_variance.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(explained_variance))],
        y=np.cumsum(explained_variance),
        mode='lines+markers',
        name='Cumulative Explained Variance',
        yaxis='y2',
        line=dict(color='red', dash='dash')
    ))
    fig_variance.update_layout(
        title='Explained Variance by Principal Components',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis2=dict(title='Cumulative Explained Variance', overlaying='y', side='right'),
        width=700,
        height=500
    )
    figs['variance'] = fig_variance
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, 
                       text_auto=True, 
                       color_continuous_scale='Blues',
                       title='Confusion Matrix',
                       labels=dict(x="Predicted Label", y="True Label"),
                       x=['Benign', 'Maligant'],
                       y=['Benign', 'Malignant'])
    fig_cm.update_layout(width=500, height=400)
    figs['confusion_matrix'] = fig_cm
    
    return figs

def generate_report(X, y, X_pca, pca_model, X_train, X_test, y_train, y_test, y_pred, accuracy, lr_model, n_components_95):
    """Generate a comprehensive analysis report"""
    report = {}
    
    # Dataset Overview
    report['dataset_overview'] = {
        'Original Shape': X.shape,
        'Reduced Shape': X_pca.shape,
        'Target Distribution': {
            'Benign': sum(y == 0),
            'Malignant': sum(y == 1)
        },
        'Components for 95% Variance': n_components_95
    }
    
    # PCA Results
    report['pca_results'] = {
        'Explained Variance Ratio (First 2)': pca_model.explained_variance_ratio_[:2].tolist(),
        'Cumulative Explained Variance (First 2)': np.cumsum(pca_model.explained_variance_ratio_[:2]).tolist(),
        'Total Explained Variance (First 2)': sum(pca_model.explained_variance_ratio_[:2])
    }
    
    # Model Performance
    report['model_performance'] = {
        'Accuracy': accuracy,
        'Classification Report': classification_report(y_test, y_pred, output_dict=True),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Logistic Regression Coefficients
    report['logistic_coefficients'] = {
        'Intercept': lr_model.intercept_.tolist(),
        'Coefficients': lr_model.coef_.flatten().tolist()
    }
    
    return report

def main():
    st.title("📊 Breast Cancer Dataset: PCA & Logistic Regression Analysis")
    st.markdown("""
    This application performs Principal Component Analysis (PCA) for dimensionality reduction 
    on the breast cancer dataset, followed by logistic regression classification. 
    The results include visualizations and performance metrics.
    """)
    
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        X, y, df = load_and_preprocess_data()
    
    # Perform PCA analysis
    with st.spinner("Performing PCA analysis..."):
        X_pca, pca_model, scaler, X_scaled, n_components_95 = perform_pca_analysis(X)
    
    # Split the data
    if X_pca.shape[1] >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Perform logistic regression
    with st.spinner("Training logistic regression model..."):
        lr_model, y_pred, accuracy = logistic_regression_analysis(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    figs = create_visualizations(df, X_pca, y, pca_model, y_test, y_pred)
    
    # Generate comprehensive report
    report = generate_report(X, y, X_pca, pca_model, X_train, X_test, y_train, y_test, y_pred, accuracy, lr_model, n_components_95)
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "PCA Analysis", "Model Performance", "Visualizations", "Detailed Report"])
    
    if page == "Dataset Overview":
        st.header("🔍 Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Features", report['dataset_overview']['Original Shape'][1])
            st.metric("Reduced Features", report['dataset_overview']['Reduced Shape'][1])
        
        with col2:
            st.metric("Total Samples", report['dataset_overview']['Original Shape'][0])
            st.metric("Components for 95% Variance", report['dataset_overview']['Components for 95% Variance'])
        
        st.subheader("Target Distribution")
        target_dist = pd.DataFrame.from_dict(report['dataset_overview']['Target Distribution'], orient='index', columns=['Count'])
        target_dist.index.name = 'Diagnosis'
        st.bar_chart(target_dist)
        
        st.subheader("First Few Rows of Dataset")
        st.dataframe(df.head())
    
    elif page == "PCA Analysis":
        st.header("🔄 Principal Component Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Explained Variance (PC1)", f"{report['pca_results']['Explained Variance Ratio (First 2)'][0]:.2%}")
        with col2:
            st.metric("Explained Variance (PC2)", f"{report['pca_results']['Explained Variance Ratio (First 2)'][1]:.2%}")
        
        st.metric("Total Explained Variance (First 2 PCs)", f"{report['pca_results']['Total Explained Variance (First 2)']:.2%}")
        
        st.subheader("Explained Variance by Components")
        if 'variance' in figs:
            st.plotly_chart(figs['variance'], use_container_width=True)
        
        st.subheader("PCA Components Information")
        components_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(pca_model.components_))],
            'Explained Variance Ratio': pca_model.explained_variance_ratio_,
            'Cumulative Variance': np.cumsum(pca_model.explained_variance_ratio_)
        })
        st.dataframe(components_df)
    
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Misclassification Rate", f"{1-accuracy:.4f}")
        
        st.subheader("Classification Report")
        cr = pd.DataFrame(report['model_performance']['Classification Report']).transpose()
        st.dataframe(cr.style.format("{:.3f}"))
        
        st.subheader("Confusion Matrix")
        if 'confusion_matrix' in figs:
            st.plotly_chart(figs['confusion_matrix'], use_container_width=True)
        
        st.subheader("Logistic Regression Coefficients")
        coef_df = pd.DataFrame({
            'Feature': [f'PC{i+1}' for i in range(len(lr_model.coef_.flatten()))],
            'Coefficient': lr_model.coef_.flatten()
        })
        st.dataframe(coef_df)
    
    elif page == "Visualizations":
        st.header("🎨 Visualizations")
        
        st.subheader("PCA: First Two Principal Components")
        if 'pca' in figs:
            st.plotly_chart(figs['pca'], use_container_width=True)
        
        st.subheader("Explained Variance by Components")
        if 'variance' in figs:
            st.plotly_chart(figs['variance'], use_container_width=True)
        
        st.subheader("Confusion Matrix")
        if 'confusion_matrix' in figs:
            st.plotly_chart(figs['confusion_matrix'], use_container_width=True)
    
    elif page == "Detailed Report":
        st.header("📋 Detailed Analysis Report")
        
        st.subheader("Dataset Information")
        st.write(f"- Original shape: {report['dataset_overview']['Original Shape']}")
        st.write(f"- Reduced shape: {report['dataset_overview']['Reduced Shape']}")
        st.write(f"- Benign samples: {report['dataset_overview']['Target Distribution']['Benign']}")
        st.write(f"- Malignant samples: {report['dataset_overview']['Target Distribution']['Malignant']}")
        st.write(f"- Components for 95% variance: {report['dataset_overview']['Components for 95% Variance']}")
        
        st.subheader("PCA Results")
        st.write(f"- Explained variance by PC1: {report['pca_results']['Explained Variance Ratio (First 2)'][0]:.2%}")
        st.write(f"- Explained variance by PC2: {report['pca_results']['Explained Variance Ratio (First 2)'][1]:.2%}")
        st.write(f"- Total explained variance (first 2 PCs): {report['pca_results']['Total Explained Variance (First 2)']:.2%}")
        
        st.subheader("Model Performance Metrics")
        st.write(f"- Accuracy: {report['model_performance']['Accuracy']:.4f}")
        st.write(f"- Precision (Benign): {report['model_performance']['Classification Report']['0']['precision']:.3f}")
        st.write(f"- Recall (Benign): {report['model_performance']['Classification Report']['0']['recall']:.3f}")
        st.write(f"- F1-Score (Benign): {report['model_performance']['Classification Report']['0']['f1-score']:.3f}")
        st.write(f"- Precision (Malignant): {report['model_performance']['Classification Report']['1']['precision']:.3f}")
        st.write(f"- Recall (Malignant): {report['model_performance']['Classification Report']['1']['recall']:.3f}")
        st.write(f"- F1-Score (Malignant): {report['model_performance']['Classification Report']['1']['f1-score']:.3f}")
        
        st.subheader("Logistic Regression Parameters")
        st.write(f"- Intercept: {report['logistic_coefficients']['Intercept']}")
        st.write(f"- Coefficients: {report['logistic_coefficients']['Coefficients']}")
        
        st.subheader("Conclusion")
        st.info(f"""
        - The PCA reduced the dataset from {report['dataset_overview']['Original Shape'][1]} features to {report['dataset_overview']['Reduced Shape'][1]} principal components
        - The first two principal components explain {report['pca_results']['Total Explained Variance (First 2)']:.2%} of the variance
        - The logistic regression model achieved an accuracy of {report['model_performance']['Accuracy']:.2%}
        - The model shows good performance on both benign and malignant classification
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Breast Cancer PCA & Logistic Regression Analysis | Created with Streamlit*")

if __name__ == "__main__":
    main()