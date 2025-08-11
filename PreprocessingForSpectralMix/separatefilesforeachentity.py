#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#MIT licence

import pandas as pd

filepath_read = "data/triples.csv"
filepath_write = "data/relations/"

triples = pd.read_csv(filepath_read)


relations = triples["relation"].unique().tolist()




for r in relations:
    print(r)
    group = triples[triples["relation"] == r]
    del group["relation"]
    del group["Unnamed: 0"]
    print(group.head())
    group.to_csv(filepath_write + r + ".csv", index=False)

print("done.")