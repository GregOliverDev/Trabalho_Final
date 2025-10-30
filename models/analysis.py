from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    target_feature = db.Column(db.String(100), nullable=False)
    creation_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)