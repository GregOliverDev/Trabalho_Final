# Trabalho_Final — Dashboard de Análises e ML (Flask)

Resumo
- Aplicação Flask para upload, armazenamento e análise de datasets CSV, com treinamento rápido de classificadores (RandomForest, DecisionTree, KNN, LogisticRegression, SVM).
- Gera gráficos (matplotlib/seaborn) e métricas (accuracy, precision, recall, f1). Usa SQLite via Flask-SQLAlchemy para registrar análises.

Requisitos
- Python 3.9+
- Windows 
- Dependências principais:
  - flask, flask_sqlalchemy
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn

Instalação
1. Criar e ativar venv:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2. Instalar dependências:
   python -m pip install --upgrade pip
   python -m pip install flask flask_sqlalchemy pandas numpy matplotlib seaborn scikit-learn

Executar
- Em desenvolvimento:
  $env:FLASK_APP = "app.py"
  flask run
  ou
  python app.py
- A aplicação inicia em http://127.0.0.1:5000 (debug=True no app.py).

Estrutura do projeto
- app.py — rotas, lógica de upload, análise, treino de modelos
- models/analysis.py — modelo SQLAlchemy (Analysis) e db
- templates/ — HTML Jinja2 (base.html, dashboard.html, create_analysis.html, analysis.html, model_results.html)
- static/css/style.css — estilos
- uploads/ — arquivos CSV enviados
- Teste/ — exemplos (winequality-red.csv, winequality-white.csv)

Rotas principais
- GET / — dashboard
- GET /analysis/<id> — página de análise; query param `feature` para análise de feature
- GET,POST /analysis/create — criar nova análise (upload CSV; campo target_feature)
- POST /analysis/delete/<id> — apagar análise
- POST /analysis/create — anexar CSV (mesmas colunas)
- GET /train_model/<id> — treinar modelo (query params para escolher classificador e hiperparâmetros)

Formato esperado do CSV
- Arquivo CSV com cabeçalho; a coluna informada em `target_feature` será a variável alvo (y). Todas as outras colunas serão usadas como X.
- Atenção: se houver colunas categóricas, o pipeline atual NÃO faz encoding automático — pode falhar em alguns classificadores.

Boas práticas e limitações
- Remove debug=True em produção; defina SECRET_KEY seguro via variável de ambiente.
- Use ambiente virtual e certifique-se que VS Code está apontando para o mesmo intérprete (Pylance).
- Para evolução do esquema do banco, usar Flask-Migrate em vez de db.create_all().
- Para datasets com features categóricas, adicionar pré-processamento (OneHotEncoder, pipelines sklearn).

Solução rápida para problemas de import
1. Ative o venv e selecione o intérprete no VS Code (Ctrl+Shift+P → Python: Select Interpreter).
2. No terminal do VS Code (com ambiente ativado), testar:
   python -c "import matplotlib, seaborn, sklearn, flask_sqlalchemy; print('OK')"
3. Instale pacotes no mesmo ambiente:
   python -m pip install matplotlib seaborn scikit-learn flask_sqlalchemy

Exemplo de uso:
1. Acesse Dashboard → Criar Nova Análise → envie CSV e informe Feature Alvo (ex: `quality`).
2. Na página da análise selecione uma feature para ver estatísticas/gráficos.
3. Em Machine Learning, escolha classificador e ajuste parâmetros (ou use padrão) → Treinar Modelo → visualizar métricas e gráfico de importância.
