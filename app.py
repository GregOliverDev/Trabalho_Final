from flask import Flask, render_template, request, redirect, url_for, flash
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px
import plotly.io as pio

from sklearn.inspection import permutation_importance
from models.analysis import Analysis, db
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# (LabelEncoder not needed currently)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)
# Use absolute path for the SQLite DB to avoid issues with relative paths
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'instance', 'analysis.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

db.init_app(app)

@app.template_filter('b64encode')
def b64encode_filter(buffer):
    # Converte buffer binário para string base64
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
    analysis = Analysis.query.get_or_404(analysis_id) # Busca análise no DB
    df = pd.read_csv(analysis.file_path) # Carrega CSV
    
    stats = {
        'total_records': len(df), # linhas
        'total_features': len(df.columns),
        'columns': df.columns.tolist()
    }
    
    # CHAMA AS DUAS FUNÇÕES, heatmap e correlacao
    heatmap_plot, correlation_matrix = generate_correlation_heatmap(df)
    target_correlations = generate_target_correlations(correlation_matrix, analysis.target_feature)
    
    if request.args.get('feature'):
        feature = request.args.get('feature')
        # carrega dados e gera gráficos
        feature_stats = generate_feature_analysis(df, feature, analysis.target_feature)  
        
        # Log p debug
        try:
            logging.info("feature=%s images keys=%s pie_none=%s bar_none=%s",
                         feature,
                         list(feature_stats['images'].keys()),
                         feature_stats['images'].get('pie_png') is None,
                         feature_stats['images'].get('bar_png') is None)
        except Exception:
            logging.exception('Failed to log feature image presence')
            
        # Retorna template COM análise específica da feature
        return render_template('analysis.html', analysis=analysis, stats=stats, feature_stats=feature_stats, heatmap_plot=heatmap_plot, target_correlations=target_correlations)
    
    # Se NÃO há parâmetro feature, mostra só estatísticas gerais
    return render_template('analysis.html', analysis=analysis, stats=stats, heatmap_plot=heatmap_plot, target_correlations=target_correlations)


# Gera análise completa de uma feature do dataset.
def generate_feature_analysis(df, feature, target_feature):
    stats = df[feature].describe().to_dict() # # Calcula count..
    correlation = None
    images = {}

    # histograma e scatter
    # Criação do gráfico em matplotlib/seaborn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # HISTOGRAMA: mostra distribuição da feature
    sns.histplot(data=df, x=feature, ax=ax1)
    ax1.set_title(f'Distribution of {feature}')

    # SCATTER PLOT, relação entre feature e target
    if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[target_feature]):
        # Calcula correlação de entre feature e target
        correlation = df[feature].corr(df[target_feature])
        sns.scatterplot(data=df, x=feature, y=target_feature, ax=ax2)
        ax2.set_title(f'{feature} vs {target_feature} (correlation: {correlation:.2f})')
    else:
        ax2.axis('off')

    # Salva a figura em buffer de memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    images['dist_scatter_png'] = buf # Armazena no dict para enviar ao template

    # GRÁFICO PIZZA: distribuição por categorias
    try:
        if pd.api.types.is_numeric_dtype(df[feature]):
            # bin numeric into categories for pie
            series_for_pie = pd.cut(df[feature], bins=5)
        else:
            series_for_pie = df[feature].astype(str)

        # Pega as 8 categorias mais frequentes
        pie_counts = series_for_pie.value_counts().nlargest(8)
        # Cria gráfico de pizza
        fig2, axp = plt.subplots(figsize=(8, 8))
        axp.pie(pie_counts.values, labels=pie_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
        axp.set_title(f'{feature} distribution')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig2)
        buf2.seek(0)
        images['pie_png'] = buf2
    except Exception:
        images['pie_png'] = None

    # GRÁFICO DE BARRAS: top valores mais frequentes
    try:
        # Conta frequência de cada valor e pega os 10 mais comuns
        bar_counts = df[feature].value_counts().nlargest(10)
        
        # Cria gráfico de barras horizonta
        fig3, axb = plt.subplots(figsize=(10, 6))
        sns.barplot(x=bar_counts.values, y=bar_counts.index.astype(str), ax=axb)
        axb.set_title(f'Top values for {feature}')
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig3)
        buf3.seek(0)
        images['bar_png'] = buf3
    except Exception:
        images['bar_png'] = None

    # GRÁFICO INTERATIVO, PLOTLY
    interactive_html = None
    try:
        if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[target_feature]):
            # Para dados numéricos: cria scatter plot interativo
            fig_px = px.scatter(df, x=feature, y=target_feature, title=f'{feature} vs {target_feature}', hover_data=df.columns)
        else:
            # Para dados categóricos: cria gráfico de barras interativo
            vc = df[feature].value_counts().nlargest(20) # Top 20 valores
            fig_px = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': feature, 'y': 'count'}, title=f'{feature} counts')

        # Converte para HTML
        interactive_html = pio.to_html(fig_px, full_html=False, include_plotlyjs='cdn')
    except Exception:
        interactive_html = None

    return {
        'stats': stats, # Estatísticas descritivas
        'correlation': correlation, # Correlação com target
        'images': images, # # Buffers das imagens
        'interactive_html': interactive_html # # HTML do gráfico Plotly
    }


# HEATMAP
def generate_correlation_heatmap(df):
    try:
        # Filtra apenas colunas numéricas
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return None, None
            
        # Calcula matriz de correlação
        correlation_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        plt.title('Mapa de Correlação entre Features Numéricas')
        plt.tight_layout()
        
        # Salva em buffer (memória) como PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        buf.seek(0)
        
        return buf, correlation_matrix # Retorna: (imagem_buffer, matriz_correlação)
    except Exception as e:
        logging.error(f"Error generating correlation heatmap: {e}")
        return None, None


def generate_target_correlations(correlation_matrix, target_feature):
    try:
        if correlation_matrix is None or target_feature not in correlation_matrix.columns:
            return None
            
        # Extrai apenas correlações com a variável target (ex: 'quality')
        target_correlations = correlation_matrix[target_feature]
        
        # Ordena em ordem decrescente (maior para menor correlação)
        sorted_correlations = target_correlations.sort_values(ascending=False)
        
        return sorted_correlations # Retorna as correlações ordenadas
    except Exception as e:
        logging.error(f"Error generating target correlations: {e}")
        return None


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
    # Run without the reloader/debugger to avoid multiple processes in background
    app.run(debug=False, host='0.0.0.0')
