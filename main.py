import code
import pandas as pd
from scipy.io import arff
data, meta = arff.loadarff(open("./data/KDDTrain+.arff"))

df = pd.DataFrame(data)

for col in [""]:
    df[col].str.decode('utf-8')

code.interact(local=locals())