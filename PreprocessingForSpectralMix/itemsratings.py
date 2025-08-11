#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#MIT licence

import pandas as pd

filepath_read = "data/ratings/recommendableItemTrue.csv"

ratings = pd.read_csv(filepath_read)

print(ratings)

print(ratings[ratings['sentiment'] == 1])