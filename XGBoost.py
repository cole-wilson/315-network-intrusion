import code
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

train_data, train_meta = arff.loadarff("./data/KDDTrain+.arff")
df_train = pd.DataFrame(train_data)
test_data, test_meta = arff.loadarff("./data/KDDTest+.arff")
df_test = pd.DataFrame(test_data)

le = LabelEncoder()
for col in df_train.columns:
    if df_train[col].dtype == object:
        df_train[col] = df_train[col].str.decode('utf-8')
        df_train[col] = le.fit_transform(df_train[col])

for col in df_test.columns:
    if df_test[col].dtype == object:
        df_test[col] = df_test[col].str.decode('utf-8')
        df_test[col] = le.fit_transform(df_test[col])

X_train = df_train.drop('class', axis=1)
y_train = df_train['class']
X_test = df_test.drop('class', axis=1)
y_test = df_test['class']

rf = XGBClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")

y_pred = rf.predict(X_test)
# Print detailed metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    # 'min_samples_split': randint(2, 10),
    # 'min_samples_leaf': randint(1, 5)
}

rf = XGBClassifier(random_state=42, n_jobs=-1)

rand_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=10, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42
)

rand_search.fit(X_train, y_train)

print(f"\nBest CV score: {rand_search.best_score_:.4f}")
print(f"Best params: {rand_search.best_params_}")

best_rf = rand_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
print(f"\nTuned Test Accuracy: {best_rf.score(X_test, y_test):.4f}")
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))