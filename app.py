from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from models.analysis import Analysis, db
import os
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

db.init_app(app)

@app.template_filter('b64encode')
def b64encode_filter(buffer):
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

with app.app_context():
    db.create_all()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def dashboard():
    analyses = Analysis.query.all()
    return render_template('dashboard.html', analyses=analyses)

@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    df = pd.read_csv(analysis.file_path)
    
    stats = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'columns': df.columns.tolist()
    }
    
    if request.args.get('feature'):
        feature = request.args.get('feature')
        feature_stats = generate_feature_analysis(df, feature, analysis.target_feature)
        return render_template('analysis.html', analysis=analysis, stats=stats, feature_stats=feature_stats)
    
    return render_template('analysis.html', analysis=analysis, stats=stats)

def generate_feature_analysis(df, feature, target_feature):
    stats = df[feature].describe().to_dict()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot histogram
    sns.histplot(data=df, x=feature, ax=ax1)
    ax1.set_title(f'Distribution of {feature}')
    
    # Calculate correlation and create scatter plot
    correlation = None
    if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[target_feature]):
        correlation = df[feature].corr(df[target_feature])
        sns.scatterplot(data=df, x=feature, y=target_feature, ax=ax2)
        ax2.set_title(f'{feature} vs {target_feature} (correlation: {correlation:.2f})')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return {
        'stats': stats,
        'correlation': correlation,
        'plot': buf
    }

@app.route('/analysis/delete/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if os.path.exists(analysis.file_path):
        os.remove(analysis.file_path)
    
    db.session.delete(analysis)
    db.session.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/analysis/create', methods=['GET', 'POST'])
def create_analysis():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = datetime.now().strftime('%Y%m%d_%H%M%S_') + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            analysis = Analysis(
                name=request.form['name'],
                file_path=file_path,
                target_feature=request.form['target_feature'],
                creation_date=datetime.now()
            )
            db.session.add(analysis)
            db.session.commit()
            
            return redirect(url_for('view_analysis', analysis_id=analysis.id))
    
    return render_template('create_analysis.html')

@app.route('/train_model/<int:analysis_id>')
def train_model(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    df = pd.read_csv(analysis.file_path)
    
    X = df.drop(analysis.target_feature, axis=1)
    y = df[analysis.target_feature]
    
    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = LabelEncoder()
    
    X_processed = X.copy()
    for column in categorical_features:
        X_processed[column] = categorical_transformer.fit_transform(X_processed[column])
    
    if not pd.api.types.is_numeric_dtype(y):
        y = categorical_transformer.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return render_template('model_results.html', metrics=metrics, plot=buf)

if __name__ == '__main__':
    app.run(debug=True)