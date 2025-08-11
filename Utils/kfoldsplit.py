
# Peter Dolog, Aalborg University, dolog@cs.aau.dk
# This was used for creating splits for mindreader dataset
# we have used splits for yelp and amazon as provided in KGCL github
import pandas as pd
import lenskit.crossfold as cf

def run():
    ratings = pd.read_csv('Data/inputfilenum_recommendableItemTrue.csv', header=None, sep=' ')
    ratings.columns = ['user', 'item', 'rating']
    #print(ratings)

    for i, tp in enumerate(cf.partition_users(ratings, 5, cf.SampleN(5))):
        tp.train.to_csv('Data/folds/train-%d.csv' % (i,),index=False)
        #tp.train.to_parquet('Data/folds/train-%d.parquet % (i,))
        tp.test.to_csv('Data/folds/test-%d.csv' % (i,),index=False)
        #tp.test.to_parquet('Data/folds/test-%d.parquet % (i,))



def main():
    run()

if __name__ == "__main__":
    main()

