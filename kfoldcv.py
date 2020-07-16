# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
#Step-1: Training data is in a CSV file called train.csv
df = pd.read_csv("train.csv")
#Step-2: create a new column called kfold and fill it with -1
df["kfold"] = -1
#Step-3: randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
#Step-4: initiate the kfold class from model_selection module
kf = model_selection.KFold(n_splits=5)
#Step-5: fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
df.loc[val_, 'kfold'] = fold
#Step-6: save the new csv with kfold column
df.to_csv("train_folds.csv", index=False)
