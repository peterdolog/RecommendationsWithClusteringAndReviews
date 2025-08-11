#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#MIT licence

import pandas as pd

filepath_read = "data/ratings.csv"
filepath_write = "data/ratings/recommendableItem"

ratings = pd.read_csv(filepath_read)


ritems = ratings["isItem"].unique().tolist()



for isitem in ritems:
    group = ratings[ratings["isItem"] == isitem]
    del group["isItem"]
    del group["Unnamed: 0"]
    group.to_csv(filepath_write + str(isitem) + ".csv", index=False)

print("done.")