#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#splits ground truth and embedding files into rating classes.
#user rows: first row like(1), second row don't know (0), third row dislike (-1)
#it also filters out items which are not in the ground truth file for each class rating separately
#from unittest.mock import inplace

import pandas as pd
import config as cfg

def run():
    print("Folds: " + str(cfg.folds))

    # This switch is valid also for LightGCN, CGCL, AdaGCL and BPRMF from LightGCN
    if cfg.KGCL:
        for i in range(0, cfg.folds):

            filepath_read_users = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                      + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + ".csv"
            filepath_read_items = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                      + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + ".csv"

            filepath_read_ground_truth = cfg.FoldsDir + "test-" + str(i) + ".csv"

            filepath_write_filterdocs = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                            + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + "filtered"
            filepath_write_filterusers = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                             + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + "filtered"

            filepath_write_truthwithoutrating = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/test-" + str(i)

                # reading truth for filtering out

            truth = pd.read_csv(filepath_read_ground_truth)
            print(filepath_read_ground_truth)


            # splitting the truth file into classes
            truth1 = truth.loc[truth["rating"] == 1]
            truth2 = truth.loc[truth["rating"] == 0]
            truth3 = truth.loc[truth["rating"] == -1]

            truth1.to_csv(filepath_write_truthwithoutrating + "-one.csv", index = False)
            truth2.to_csv(filepath_write_truthwithoutrating + "-zero.csv", index = False)
            truth3.to_csv(filepath_write_truthwithoutrating + "-minusone.csv", index = False)

            # for spectralmix, the delimiter is space
            #docs = pd.read_csv(filepath_read_items, header=None, sep=" ")
            # for spectral clustering the delimiter is the colon
            docs = pd.read_csv(filepath_read_items, header=None, sep=",")

            docs.rename(columns={0: "item"}, inplace=True)

            if cfg.ReviewsOnly:
                itemlistfile = "item_list.txt"
                userlistfile = "user_list.txt"

                itemListDF = pd.read_csv(cfg.SentenceEmbDir + itemlistfile, sep=' ')
                userListDF = pd.read_csv(cfg.SentenceEmbDir + userlistfile, sep=' ')

                maxItemID = itemListDF["remap_id"].max()
                maxUserID = userListDF["remap_id"].max()

                offsetForItems = maxUserID + 1
                offsetForEntities = offsetForItems + maxItemID + 1

                docs['item'] =  docs['item'] - offsetForItems



            docs.rename(columns={0: "item"}, inplace=True)
            truth1_for_merge = pd.DataFrame(truth1['item'])
            truth1_for_merge.drop_duplicates(subset=['item'], keep='first', inplace=True)
            truth1_for_merge.reset_index(inplace=True, drop=True)
            docs_filtered1 = pd.merge(truth1_for_merge, docs, how="inner", on="item")
            docs_filtered2 = pd.merge(truth2, docs, how="inner", on="item")
            docs_filtered3 = pd.merge(truth3, docs, how="inner", on="item")
            docs_filtered2.drop('user', inplace=True, axis=1)
            docs_filtered2.drop('rating', inplace=True, axis=1)
            docs_filtered3.drop('user', inplace=True, axis=1)
            docs_filtered3.drop('rating', inplace=True, axis=1)


            docs_filtered_nodupl1 = docs_filtered1.drop_duplicates(keep='first')
            docs_filtered_nodupl2 = docs_filtered2.drop_duplicates(keep='first')
            docs_filtered_nodupl3 = docs_filtered3.drop_duplicates(keep='first')

            docs_filtered_nodupl1.to_csv(filepath_write_filterdocs + "class1.csv", index=False, header=False)
            docs_filtered_nodupl2.to_csv(filepath_write_filterdocs + "class2.csv", index=False, header=False)
            docs_filtered_nodupl3.to_csv(filepath_write_filterdocs + "class3.csv", index=False, header=False)

            print("writing 3 filtered classes for documents according to truth files done.")

            # keeping the style that the user preference for positive rating is in the first row
            # for spectral clustering we do not differentiate but for the rest of the processing the file should stay
            # named the same
            users = pd.read_csv(filepath_read_users, header=None, sep=",")
            users.to_csv(filepath_write_filterusers + "1row.csv", index=False, header=False)

    elif cfg.spectralclustering:
        for i in range(0, cfg.folds):

            filepath_read_users = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                      + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + ".csv"
            filepath_read_items = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                      + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + ".csv"

            filepath_read_ground_truth = cfg.FoldsDir + "test-" + str(i) + ".csv"

            filepath_write_filterdocs = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                            + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + "filtered"
            filepath_write_filterusers = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                             + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(
                    cfg.seed) + "filtered"

            filepath_write_truthwithoutrating = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/test-" + str(i)

            # reading truth for filtering out

            truth = pd.read_csv(filepath_read_ground_truth)

            # splitting the truth file into classes
            truth1 = truth.loc[truth["rating"] == 1]
            truth2 = truth.loc[truth["rating"] == 0]
            truth3 = truth.loc[truth["rating"] == -1]

            truth1.to_csv(filepath_write_truthwithoutrating + "-one.csv", index = False)
            truth2.to_csv(filepath_write_truthwithoutrating + "-zero.csv", index = False)
            truth3.to_csv(filepath_write_truthwithoutrating + "-minusone.csv", index = False)

            print("Writing three classes for truth file done.")

            docs = pd.read_csv(filepath_read_items, header=None, sep=",")
            docs.rename(columns={0: "item"}, inplace=True)

            docs_filtered1 = pd.merge(truth1, docs, how="inner", on="item")
            docs_filtered2 = pd.merge(truth2, docs, how="inner", on="item")
            docs_filtered3 = pd.merge(truth3, docs, how="inner", on="item")

            docs_filtered1.drop('user', inplace=True, axis=1)
            docs_filtered1.drop('rating', inplace=True, axis=1)
            docs_filtered2.drop('user', inplace=True, axis=1)
            docs_filtered2.drop('rating', inplace=True, axis=1)
            docs_filtered3.drop('user', inplace=True, axis=1)
            docs_filtered3.drop('rating', inplace=True, axis=1)

            docs_filtered_nodupl1 = docs_filtered1.drop_duplicates(keep='first')
            docs_filtered_nodupl2 = docs_filtered2.drop_duplicates(keep='first')
            docs_filtered_nodupl3 = docs_filtered1.drop_duplicates(keep='first')

            docs_filtered_nodupl1.to_csv(filepath_write_filterdocs + "class1.csv", index=False, header=False)
            docs_filtered_nodupl2.to_csv(filepath_write_filterdocs + "class2.csv", index=False, header=False)
            docs_filtered_nodupl3.to_csv(filepath_write_filterdocs + "class3.csv", index=False, header=False)

            print("writing 3 filtered classes for documents according to truth files done.")
            users = pd.read_csv(filepath_read_users, header=None, sep=",")
            # keeping the style that the user preference for positive rating is in the first row
            # for spectral clustering we do not differentiate but for the rest of the processing the file should stay
            # named the same
            users.to_csv(filepath_write_filterusers + "1row.csv", index=False, header=False)

    elif cfg.SpectralMix:
        for i in range(0, cfg.folds):

            filepath_read_users = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                  + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + ".csv"
            filepath_read_items = cfg.SpectralMixEmbeddingsDir + "seed" + str(cfg.seed) \
                                  + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + ".csv"


            filepath_read_ground_truth = cfg.FoldsDir + "test-" + str(i) + ".csv"

            filepath_write_filterdocs = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                        + "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filtered"
            filepath_write_filterusers = cfg.SplitToClassDir + "seed" + str(cfg.seed) \
                                         + "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filtered"

            filepath_write_truthwithoutrating = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/test-" + str(i)

            # reading truth for filtering out

            truth = pd.read_csv(filepath_read_ground_truth)
            print(filepath_read_ground_truth)

            # splitting the truth file into classes
            truth1 = truth.loc[truth["rating"] == 1]
            truth2 = truth.loc[truth["rating"] == 0]
            truth3 = truth.loc[truth["rating"] == -1]


            truth1.to_csv(filepath_write_truthwithoutrating + "-one.csv", index = False)
            truth2.to_csv(filepath_write_truthwithoutrating + "-zero.csv", index = False)
            truth3.to_csv(filepath_write_truthwithoutrating + "-minusone.csv", index = False)

            print("Writing three classes for truth file done.")

            # for spectralmix, the delimiter is space
            docs = pd.read_csv(filepath_read_items, header=None, sep=" ")
            #for spectral clustering the delimiter is the colon
            #docs = pd.read_csv(filepath_read_items, header=None, sep=",")
            docs.rename(columns={0: "item"}, inplace=True)

            docs_filtered1 = pd.merge(truth1, docs, how="inner", on="item")
            docs_filtered2 = pd.merge(truth2, docs, how="inner", on="item")
            docs_filtered3 = pd.merge(truth3, docs, how="inner", on="item")

            docs_filtered1.drop('user', inplace=True, axis=1)
            docs_filtered1.drop('rating', inplace=True, axis=1)
            docs_filtered2.drop('user', inplace=True, axis=1)
            docs_filtered2.drop('rating', inplace=True, axis=1)
            docs_filtered3.drop('user', inplace=True, axis=1)
            docs_filtered3.drop('rating', inplace=True, axis=1)


            docs_filtered_nodupl1 = docs_filtered1.drop_duplicates(keep='first')
            docs_filtered_nodupl2 = docs_filtered2.drop_duplicates(keep='first')
            docs_filtered_nodupl3 = docs_filtered1.drop_duplicates(keep='first')
            docs_filtered_nodupl1.to_csv(filepath_write_filterdocs + "class1.csv", index=False, header=False)
            docs_filtered_nodupl2.to_csv(filepath_write_filterdocs + "class2.csv", index=False, header=False)
            docs_filtered_nodupl3.to_csv(filepath_write_filterdocs + "class3.csv", index=False, header=False)

            print("writing 3 filtered classes for documents according to truth files done.")

            # filtering users
            # first row -1, the second row 0, the third row 1
            #for spectral mix
            users = pd.read_csv(filepath_read_users, header=None, sep=" ")
            #for spectral clustering
            #users = pd.read_csv(filepath_read_users, header=None, sep=",")
            # first emb for a user rating is like, the second is not interested, and the last is dislike
            preference = 0
            user1row = pd.DataFrame()
            user2row = pd.DataFrame()
            user3row = pd.DataFrame()
            for index, row in users.iterrows():
                print("processing user preference: " + str(preference))
                print("processing user row: " + str(index))
                if preference == 0:
                    preference = preference + 1
                elif preference == 1:
                    print("if 1 preference " + str(preference))
                    user1row = user1row.append(row, ignore_index=True)
                    preference = preference + 1
                elif preference == 2:
                    print("if 2 preference " + str(preference))
                    user2row = user2row.append(row, ignore_index=True)
                    preference = preference + 1
                elif preference == 3:
                    print("if 3 preference " + str(preference))
                    user3row = user3row.append(row, ignore_index=True)
                    preference = 0
                # if preference == 4:
                    # print("if 4")
                    #   preference = 0
                    print("At the end if " + str(preference))

                print("users")
                print(len(users))
                print("class 1:")
                print(len(user1row))
                print("class 2:")
                print(len(user2row))
                print("class 3:")
                print(len(user3row))

                user1row.to_csv(filepath_write_filterusers + "1row.csv", index=False, header=False)
                user2row.to_csv(filepath_write_filterusers + "2row.csv", index=False, header=False)
                user3row.to_csv(filepath_write_filterusers + "3row.csv", index=False, header=False)

                print("Writing three classes for users done.")


def main():
    run()


if __name__ == "__main__":
    main()