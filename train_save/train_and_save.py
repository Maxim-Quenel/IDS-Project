# train_and_save.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ---- CONFIG ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_FILENAMES = ["KDDTrain+.txt", "KDDTest+.txt"]

def resolve_dataset_paths():
    search_dirs = [SCRIPT_DIR, PARENT_DIR]
    for search_dir in search_dirs:
        train_candidate = os.path.join(search_dir, "KDDTrain+.txt")
        test_candidate = os.path.join(search_dir, "KDDTest+.txt")
        if os.path.isfile(train_candidate) and os.path.isfile(test_candidate):
            return train_candidate, test_candidate
    searched = ", ".join(search_dirs)
    raise FileNotFoundError(
        f"Dataset files not found. Looked in: {searched}. Expected KDDTrain+.txt and KDDTest+.txt."
    )

train_file, test_file = resolve_dataset_paths()
OUTPUT_DIR = PARENT_DIR

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

# ---- Chargement ----
df_train = pd.read_csv(train_file, names=columns)
df_test = pd.read_csv(test_file, names=columns)

# ---- Prétraitement / étiquetage binaire ----
def prepare_df(df):
    df = df.copy()
    # conserver les colonnes utiles ; retirer 'difficulty' qui n'est pas info réseau
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
    # convertir label en binaire : normal -> 0, attack -> 1
    df['label_bin'] = df['label'].apply(lambda x: 0 if x.strip().lower() == 'normal' else 1)
    # certaines colonnes peuvent être non numériques : check
    return df

df_train = prepare_df(df_train)
df_test = prepare_df(df_test)

# Séparer X / y
feature_cols = [c for c in df_train.columns if c not in ('label', 'label_bin')]
X_train = df_train[feature_cols]
y_train = df_train['label_bin'].astype(int)

X_test = df_test[feature_cols]
y_test = df_test['label_bin'].astype(int)

# ---- Détection colonnes catégorielles / numériques ----
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

# ---- Pipeline de prétraitement ----
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ],
    remainder='drop'
)

# ---- Pipeline complet ----
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', clf)
])

# ---- Option : recherche d'hyperparamètres (facultative, ici petit grid) ----
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 20],
}

grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

print("Training... (this may take a few minutes)")
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
print("Best params:", grid.best_params_)

# ---- Évaluation sur test ----
y_pred = best_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---- Sauvegarde ----
model_path = os.path.join(OUTPUT_DIR, "model_pipeline.joblib")
joblib.dump({
    'pipeline': best_pipeline,
    'feature_cols': feature_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols
}, model_path)

print(f"Model saved to {model_path}")
