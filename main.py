import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_ids = test_df['id'].copy()

train_df['is_train'] = 1
test_df['is_train'] = 0
combined = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)

target = 'Status'

drop_cols = ['id', 'is_train']
if target in combined.columns:
    drop_cols.append(target)

features = [col for col in combined.columns if col not in drop_cols]

numeric_features = combined[features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = combined[features].select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

combined_processed = preprocessor.fit_transform(combined[features])

cat_feature_names = []
try:
    for i, col in enumerate(categorical_features):
        cats = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
        cat_feature_names.extend([f"{col}_{cat}" for cat in cats])
except:
    cat_feature_names = [f"cat_{i}" for i in range(combined_processed.shape[1] - len(numeric_features))]

feature_names = numeric_features + cat_feature_names

combined_processed_df = pd.DataFrame(combined_processed, columns=feature_names[:combined_processed.shape[1]])

train_indices = combined[combined['is_train'] == 1].index
test_indices = combined[combined['is_train'] == 0].index

train_processed = combined_processed_df.loc[train_indices].reset_index(drop=True)
test_processed = combined_processed_df.loc[test_indices].reset_index(drop=True)

y_train = train_df[target].values

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(
    train_processed, 
    y_train_encoded,
    eval_set=[(train_processed, y_train_encoded)],
    verbose=False
)

y_pred_proba = model.predict_proba(test_processed)

submission = pd.DataFrame({
    'id': test_ids,
    'Status_C': y_pred_proba[:, 0],
    'Status_CL': y_pred_proba[:, 1],
    'Status_D': y_pred_proba[:, 2]
})

submission.to_csv('submission.csv', index=False)
