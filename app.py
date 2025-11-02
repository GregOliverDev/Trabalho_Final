from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.inspection import permutation_importance
from models.analysis import Analysis, db
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(data=df, x=feature, ax=ax1)
    ax1.set_title(f'Distribution of {feature}')
    
    correlation = None
    if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[target_feature]):
        correlation = df[feature].corr(df[target_feature])
        sns.scatterplot(data=df, x=feature, y=target_feature, ax=ax2)
        ax2.set_title(f'{feature} vs {target_feature} (correlation: {correlation:.2f})')
    
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
    
    clf_name = request.args.get('classifier', request.form.get('classifier', 'random_forest')).lower()

    model = None
    model_params = {}

    if clf_name in ['random_forest', 'rf']:
        n_estimators = int(request.args.get('n_estimators', 100))
        max_depth = int(request.args.get('max_depth')) if request.args.get('max_depth') else None
        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42}
        model = RandomForestClassifier(**{k:v for k,v in model_params.items() if v is not None})

    elif clf_name in ['decision_tree', 'dt']:
        max_depth = int(request.args.get('max_depth')) if request.args.get('max_depth') else None
        model_params = {'max_depth': max_depth, 'random_state': 42}
        model = DecisionTreeClassifier(**{k:v for k,v in model_params.items() if v is not None})

    elif clf_name in ['knn', 'k_neighbors', 'knearest']:
        n_neighbors = int(request.args.get('n_neighbors', 5))
        model_params = {'n_neighbors': n_neighbors}
        model = KNeighborsClassifier(**model_params)

    elif clf_name in ['logistic', 'logistic_regression', 'lr']:
        C = float(request.args.get('C', 1.0))
        max_iter = int(request.args.get('max_iter', 100))
        model_params = {'C': C, 'max_iter': max_iter, 'random_state': 42}
        model = LogisticRegression(**model_params)

    elif clf_name in ['svm', 'svc']:
        C = float(request.args.get('C', 1.0))
        kernel = request.args.get('kernel', 'rbf')
        model_params = {'C': C, 'kernel': kernel, 'random_state': 42}
        model = SVC(**model_params)

    else:
        # Random Forest como modelo default, caso nenhum seja especificado
        model = RandomForestClassifier(random_state=42)
        model_params = {'n_estimators': 100, 'random_state': 42}
        clf_name = 'random_forest'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Nem todos os modelos tem feature_importances_, usando permutation importance como alternativa
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Feature Importance ({clf_name})')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return render_template('model_results.html', metrics=metrics, plot=buf, model_name=clf_name, model_params=model_params)


@app.route('/analysis/<int:analysis_id>/append', methods=['POST'])
def append_data(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)

    if 'file' not in request.files:
        flash('Nenhum arquivo enviado', 'warning')
        return redirect(url_for('view_analysis', analysis_id=analysis.id))

    file = request.files['file']
    if file.filename == '':
        flash('Nenhum arquivo selecionado', 'warning')
        return redirect(url_for('view_analysis', analysis_id=analysis.id))

    try:
        df_existing = pd.read_csv(analysis.file_path)
        new_df = pd.read_csv(file)
    except Exception as e:
        flash(f'Erro ao ler o arquivo CSV: {e}', 'warning')
        return redirect(url_for('view_analysis', analysis_id=analysis.id))

    if set(df_existing.columns) != set(new_df.columns):
        flash('dataset com colunas diferentes das usadas na criação da analise', 'warning')
        return redirect(url_for('view_analysis', analysis_id=analysis.id))

    new_df = new_df[df_existing.columns]

    try:
        new_df.to_csv(analysis.file_path, mode='a', header=False, index=False)
    except Exception as e:
        flash(f'Erro ao salvar os dados: {e}', 'warning')
        return redirect(url_for('view_analysis', analysis_id=analysis.id))

    flash('Dados inseridos com sucesso', 'success')
    return redirect(url_for('view_analysis', analysis_id=analysis.id))

if __name__ == '__main__':
    app.run(debug=True)
