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
import streamlit as st

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
                       x=['Benign', 'Malignant'],
                       y=['Benign', 'Malignant'])
    fig_cm.update_layout(width=500, height=400)
    figs['confusion_matrix'] = fig_cm
    
    # Feature Importance (Coefficients from Logistic Regression)
    feature_importance = pd.DataFrame({
        'Feature': ['PC1', 'PC2'][:len(X_pca[0])],
        'Coefficient': [0, 0][:len(X_pca[0])]  # Placeholder - will be updated later
    })
    
    fig_importance = px.bar(feature_importance, 
                            x='Feature', 
                            y='Coefficient',
                            title='Logistic Regression Coefficients (PCA Features)',
                            color='Coefficient',
                            color_continuous_scale='RdBu')
    fig_importance.update_layout(width=600, height=400)
    figs['importance'] = fig_importance
    
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

def main_analysis():
    """Main analysis function"""
    # Load and preprocess data
    X, y, df = load_and_preprocess_data()
    
    # Perform PCA analysis
    X_pca, pca_model, scaler, X_scaled, n_components_95 = perform_pca_analysis(X)
    
    # Split the data
    if X_pca.shape[1] >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Perform logistic regression
    lr_model, y_pred, accuracy = logistic_regression_analysis(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    figs = create_visualizations(df, X_pca, y, pca_model, y_test, y_pred)
    
    # Update importance chart with actual coefficients
    feature_importance = pd.DataFrame({
        'Feature': [f'PC{i+1}' for i in range(len(lr_model.coef_.flatten()))],
        'Coefficient': lr_model.coef_.flatten()
    })
    
    fig_importance = px.bar(feature_importance, 
                            x='Feature', 
                            y='Coefficient',
                            title='Logistic Regression Coefficients (PCA Features)',
                            color='Coefficient',
                            color_continuous_scale='RdBu')
    fig_importance.update_layout(width=600, height=400)
    figs['importance'] = fig_importance
    
    # Generate comprehensive report
    report = generate_report(X, y, X_pca, pca_model, X_train, X_test, y_train, y_test, y_pred, accuracy, lr_model, n_components_95)
    
    return X, y, X_pca, pca_model, X_train, X_test, y_train, y_test, y_pred, accuracy, lr_model, figs, report, scaler

if __name__ == "__main__":
    # Run the analysis
    X, y, X_pca, pca_model, X_train, X_test, y_train, y_test, y_pred, accuracy, lr_model, figs, report, scaler = main_analysis()
    
    # Print summary
    print("=== PCA and Logistic Regression Analysis Summary ===")
    print(f"Original dataset shape: {report['dataset_overview']['Original Shape']}")
    print(f"Reduced dataset shape: {report['dataset_overview']['Reduced Shape']}")
    print(f"Components needed for 95% variance: {report['dataset_overview']['Components for 95% Variance']}")
    print(f"Total explained variance (first 2 PCs): {report['pca_results']['Total Explained Variance (First 2)']:.2%}")
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))