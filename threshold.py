# https://ieeexplore.ieee.org/document/6009578 -- An excellent reference for clustering-based binary classification

# https://www.researchgate.net/publication/271891551_Network_Anomaly_Detection_by_Cascading_K-Means_Clustering_and_C45_Decision_Tree_algorithm 
# 
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)

DATA_DIR     = "./data"
TRAIN_FILE   = "KDDTrain+.txt"
TEST_FILE    = "KDDTest+.txt"
N_CLUSTERS   = 9
RANDOM_STATE = 42
N_TREES      = 600
MAX_DEPTH    = 20

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "level",
]

CATEGORIES = {
    "DoS":   ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
              "udpstorm", "processtable", "mailbomb", "worm"],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L":   ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
              "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
              "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
              "named", "worm"],
    "U2R":   ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
              "sqlattack", "xterm"],
    "Normal": ["normal"],
}

#  Returns the category for a given attack name
def categorize(attack_name):
    for cat, attacks in CATEGORIES.items():
        if attack_name in attacks:
            return cat
    return "other"


# Load data
df_train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), names=COLUMNS)
df_test  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE),  names=COLUMNS)

# Binary label: 0 = normal, 1 = attack
df_train["is_attack"] = (df_train["attack"] != "normal").astype(int)
df_test["is_attack"] = (df_test["attack"]  != "normal").astype(int)


df_test["category"] = df_test["attack"].apply(categorize)


for df in (df_train, df_test):
    if "num_outbound_cmds" in df.columns:
        df.drop(columns="num_outbound_cmds", inplace=True)

drop_cols = ["attack", "level", "is_attack", "category"]
cat_cols  = ["protocol_type", "service", "flag"]

excluded_cols = set(drop_cols + cat_cols)
num_cols = [col for col in df_train.columns if col not in excluded_cols]

# https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(df_train[cat_cols])
X_test_cat  = ohe.transform(df_test[cat_cols])

X_train = np.hstack([df_train[num_cols].astype(np.float32).values, X_train_cat.astype(np.float32)])
X_test  = np.hstack([df_test[num_cols].astype(np.float32).values, X_test_cat.astype(np.float32)])


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html -- Transform features by scaling each feature to a given range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

y_train = df_train["is_attack"].values
y_test  = df_test["is_attack"].values

print(f"Clustering into {N_CLUSTERS} groups")
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
    max_iter=500, n_init=10, batch_size=4096,
)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters  = kmeans.predict(X_test_scaled)

cluster = {}
for cluster_id in range(N_CLUSTERS):
    labels = y_train[train_clusters == cluster_id]
    total = len(labels)
    cluster_attack_rate = 100 * np.mean(labels == 1) if total else 0.0

    mapping = (
        1 if total <= 50 else
        0 if cluster_attack_rate < 0.01 else
        1 if cluster_attack_rate > 99.9 else
        None
    )

    cluster[cluster_id] = {
        "mapping": mapping,
        "size": int(total),
        "cluster_attack_rate": float(cluster_attack_rate),
    }

print("training per cluster")
cluster_models = {}
for cluster_id, info in cluster.items():
    if info["mapping"] is not None:
        continue
    mask = train_clusters == cluster_id

    rf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH,
        max_features="sqrt", min_samples_split=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_scaled[mask], y_train[mask])
    cluster_models[cluster_id] = rf
    print(f"Cluster {cluster_id}: trained (size={info['size']:,}, "f"attack%={info['cluster_attack_rate']:.1f}%)")

test_probability = np.zeros(len(X_test_scaled))
for cluster_id in np.unique(test_clusters):
    cluster_indices = np.where(test_clusters == cluster_id)[0]
    cluster_data = X_test_scaled[cluster_indices]
    cluster_mapping = cluster[cluster_id]["mapping"]

    if cluster_mapping is not None:
        test_probability[cluster_indices] = float(cluster_mapping)
    else:
        probabilities = cluster_models[cluster_id].predict_proba(cluster_data)
        test_probability[cluster_indices] = probabilities[:, 1]


# Prints a specific thresholds metrics
def report(threshold):
    y_pred = (test_probability >= threshold).astype(int)
    cm  = confusion_matrix(y_test, y_pred)  
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    detection_rate  = tp / (tp + fn)
    false_alarm_rate = fp / (fp + tn)

    print("\n" + "=" * 60)
    print(f"RESULTS {threshold}")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"F1 Score:         {f1:.4f}")
    print(f"Detection Rate:   {detection_rate * 100:.2f}%  ({tp:,}/{tp + fn:,} caught)")
    print(f"False Alarm Rate: {false_alarm_rate * 100:.2f}%  ({fp:,})")

    print("\nConfusion matrix:")
    print(f"                  Predicted Normal    Predicted Attack")
    print(f"  Actual Normal     {cm[0,0]}    {cm[0,1]}")
    print(f"  Actual Attack     {cm[1,0]}    {cm[1,1]}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    print("per cat recall:")
    df_diag = pd.DataFrame({
        "category": df_test["category"].values,
        "y_true":   y_test,
        "y_pred":   y_pred,
    })
    for cat in ["DoS", "Probe", "R2L", "U2R"]:
        sub = df_diag[df_diag["category"] == cat]
        if len(sub) == 0:
            continue
        n_caught = int(((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum())
        print(f"{cat:>6}: {n_caught / len(sub):.4f}  ({n_caught:,}/{len(sub):,})")


report(0.5)

# Here we use a very aggressive threshold, which results in a much higher detection rate with the cost being more false alarms.
report(0.02)

