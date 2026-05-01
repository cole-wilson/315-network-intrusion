import pandas as pd
import os
from scipy.io import arff


DATA_DIR     = "./data"
TRAIN_FILE   = "KDDTrain+.txt"
TEST_FILE    = "KDDTest+.txt"

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

df_train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), names=COLUMNS)
df_test  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE),  names=COLUMNS)
train = set(df_train["attack"])
test = set(df_test["attack"])
print("train", train)
print("test", test)
print()
print()
print()
print("train&test", train&test, len(train&test))
print()
print("train-test", train-test, len(train-test))
print()
print("test-train", test-train, len(test-train))