# Peter Dolog, Aalborg University, dolog@cs.aau.dk
# various functions to help producing plots and tables for the paper
# e.g. aggregating evaluations from all methods, creating subrables, creating different plots for differnt measures, selecting parts of evaluation results and so on


from pickle import FALSE

import pandas as pd
from pandas import read_csv

import config as cfg
import os
import matplotlib.pyplot as plt


def get_eval_dataframe(path):
    df = pd.read_csv(path)
    return df

def concat_dfs(df1, df2):
    df1 = pd.concat([df1, df2])
    return df1

def change_separator_from_semicolumn_to_comma(filepath):
    df = pd.read_csv(filepath, sep=';')
    df.to_csv(filepath, sep=",", index=False)

def remove_columns(filepath, p_columns):
    df = pd.read_csv(filepath, sep=';')
    if set(p_columns).issubset(df.columns):
        df.drop(p_columns, axis=1, inplace=True)
    df.to_csv(filepath + "_removed", sep=";", index=False)

def CreatedAggregatedResultswithDistinguishingAlgorithm(dataset, sort_column):
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"

    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/withAdaCGLandCGCL/"

    directory = "Data/eval/5FoldEval/AggregatedEvalYelp/aggregations/"
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/AdditionsofAdaCGLandCGCL/"
    #change_separator_from_semicolumn_to_comma(directory + "AggAll_Amazon_ndcg_5_withoutAdaCGLandCGCL.csv")



    #sort_column = 'hit_5'
    #write_file = "AggregatedAll_amazon_agg" + "_" + "sortedby_" + sort_column + ".csv"
    #write_file = "AggregatedAll_amazon_agg" + "_" + sort_column + ".csv"
    write_file = "AggregatedAll_" + dataset + "_" + sort_column + ".csv"


    files = [f for f in os.listdir(directory)]

    dfs = []
    filenr = 0
    for f in files:
        filepath = directory + f
        fsplit = f.split('_')
        filenr = filenr + 1
        if f != write_file and fsplit[0] != "AggregatedAll":
            df = pd.read_csv(filepath, sep=',')
            if set(['nrecs', 'ndcg_un', 'hit_un', 'P_un', 'R_un', 'RR_un']).issubset(df.columns):
                df.drop(['nrecs', 'ndcg_un', 'hit_un', 'P_un', 'R_un', 'RR_un'], axis=1, inplace=True)

            print(df)
            baselines1 = df[df['Algorithm'] == 'AlternateLeastSquares']
            baselines2 = df[df['Algorithm'] == 'UserKNN']
            baselines = pd.concat([baselines1, baselines2], ignore_index=True)




            if 'BPR' in set(df['Algorithm']):
                df = df.drop(df.loc[df['Algorithm']=='BPR'].index)


            if 'UserKNN' in set(df['Algorithm']):
                df = df.drop(df.loc[df['Algorithm']=='UserKNN'].index)

            if 'AlternateLeastSquares' in set(df['Algorithm']):
                df = df.drop(df.loc[df['Algorithm']=='AlternateLeastSquares'].index)

            df['Algorithm'] = df['Algorithm'].str.replace('dotproduct', 'DP')
            df['Algorithm'] = df['Algorithm'].str.replace('euclidean', 'E')
            df['Algorithm'] = df['Algorithm'].str.replace('reviews', 'R')
            df['Algorithm'] = df['Algorithm'].str.replace('dotclusters', 'DC')
            df['Algorithm'] = df['Algorithm'].str.replace('concatclusters', 'CC')
            df['Algorithm'] = df['Algorithm'].str.replace('seed', 'S')
            df['Algorithm'] = df['Algorithm'].str.replace('2clusters', '2C')
            df['Algorithm'] = df['Algorithm'].str.replace('3clusters', '3C')
            df['Algorithm'] = df['Algorithm'].str.replace('4clusters', '4C')
            df['Algorithm'] = df['Algorithm'].str.replace('5clusters', '5C')
            df['Algorithm'] = df['Algorithm'].str.replace('movieRatings', 'MR')
            df['Algorithm'] = df['Algorithm'].str.replace('allRatings', 'AR')


            df['Algorithm'] = fsplit[0] + "_" + df['Algorithm']

            df['Algorithm'] = df['Algorithm'].str.replace('movieRatings', 'MR')
            df['Algorithm'] = df['Algorithm'].str.replace('movieratings', 'MR')
            df['Algorithm'] = df['Algorithm'].str.replace('allRatings', 'AR')
            df['Algorithm'] = df['Algorithm'].str.replace('allratings', 'AR')
            df['Algorithm'] = df['Algorithm'].str.replace('seed', 'S')
            df['Algorithm'] = df['Algorithm'].str.replace('Seed', 'S')

            df['Algorithm'] = df['Algorithm'].str.replace('CosineTimesEuclidean', 'OPDPxED')

            df['Algorithm'] = df['Algorithm'].str.replace('Cosine', 'OFDP')

            df['Algorithm'] = df['Algorithm'].str.replace('_and', '')
            df['Algorithm'] = df['Algorithm'].str.replace('_', '-')

            dfs.append(df)

    dfa = pd.concat(dfs, ignore_index=True)
    dfa = pd.concat([baselines, dfa], ignore_index=True)


    dfa = dfa.sort_values(sort_column, ascending=False)




    dfa.to_csv(directory + write_file, sep=";", index=False)


    # if some evaluations use additional columns

    p_columns = ['ndcg_20', 'hit_20', 'P_20', 'R_20', 'RR_20']
    remove_columns(directory + write_file, p_columns)

    # if some prefix is added

    #dfa = pd.read_csv(directory + write_file + "_removed", sep=';')

    dfa = pd.read_csv(directory + write_file + "_removed", sep=';')
    #dfa = pd.read_csv(directory + write_file, sep=';')

    p_prefix = "AggAll-"



    dfa['Algorithm'] = dfa['Algorithm'].str.replace(p_prefix, '')
    print(dfa)
    dfa.to_csv(directory + write_file + '_removeagg', sep=";", index=False)

    return dfa




def remove_agg_label(path):
    df = pd.read_csv(path, sep=';')
    p_prefix = "AggAll-"
    df['Algorithm'] = df['Algorithm'].str.replace(p_prefix, '')
    df.to_csv(path, sep=";", index=False)


def format_panda_to_latex_table(dfa, dataset, sort_column):
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    directory = "Data/eval/5FoldEval/AggregatedEvalYelp/"
    write_file_tex = "AggregatedAll_" + dataset + "_" + sort_column + ".tex"

    #latex cannot deal with "_" character. Must be espaced as follows: "\_"
    #dfa['Algorithm'] = dfa['Algorithm'].str.replace('_', '-')
    dfa.columns = dfa.columns.astype(str).str.replace("_", "-")
    dfa = dfa.round(4)


    s = dfa.style.highlight_max(
        props='cellcolor:[HTML]{FFFF00}; color:{red};'
              'textit:--rwrap; textbf:--rwrap;'
    )
    s.to_latex(buf=directory + write_file_tex)

def format_panda_to_latex_table_fromfile(dataset, sort_column):
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    directory = "Data/eval/5FoldEval/AggregatedEvalYelp/"
    read_csv_file = "AggregatedAll_" + dataset + "_" + sort_column + ".csv"
    write_file_tex = "AggregatedAll_" + dataset + "_" + sort_column + ".tex"

    dfa = pd.read_csv(directory + read_csv_file, sep=';')

    #latex cannot deal with "_" character. Must be espaced as follows: "\_"
    #dfa['Algorithm'] = dfa['Algorithm'].str.replace('_', '-')
    dfa.columns = dfa.columns.astype(str).str.replace("_", "-")
    dfa = dfa.round(4)


    s = dfa.style.highlight_max(
        props='cellcolor:[HTML]{FFFF00}; color:{red};'
              'textit:--rwrap; textbf:--rwrap;'
    )
    s.to_latex(buf=directory + write_file_tex)

def AggregatePerEvaluation(algorithm, dataset):
    directory = "Data/eval/5FoldEval/AggregatedEvalMindreader"
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    write_file = algorithm + "_" + dataset + "_agg.csv"
    files = [f for f in os.listdir(directory)]
    # print(files)
    dfs = []
    for f in files:
        filepath = directory + f
        if f != write_file:
            df = pd.read_csv(filepath)
            print(df)
            df['Algorithm'] = algorithm + "_" + df['Algorithm']
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(df)
    df.to_csv(directory + write_file, sep=";", index=False)

def give_best_performing_per_column(df, dataset):
    best_performing = {}
    for col in df.columns:
        if col not in ['Algorithm', 'nrecs']:
            #max_val = df[col].max()
            df2 = df.nlargest(1, [col])
            #print(col)
            #print(df2['Algorithm'])
            #df2 = df.query(f'Fee == {max_val}')

            alg = str(df2['Algorithm'].iloc[0])
            best_performing[col] = alg

            #print(best_performing)

def give_best_performing_per_column_asdf(df, dataset):
    #best_performing = {}

    best_performing_agg = pd.DataFrame()
    for col in df.columns:
        if col not in ['Algorithm', 'nrecs']:
            best_performing = pd.DataFrame()
            # max_val = df[col].max()
            df2 = df.nlargest(3, [col])
            # print(col)
            # print(df2['Algorithm'])
            # df2 = df.query(f'Fee == {max_val}')
            #print(df2[['Algorithm', col]])
            best_performing = df2[['Algorithm', col]]
            best_performing['Measure'] = col
            best_performing = best_performing.rename(columns={col : "Value"})
            #print(best_performing)
            #exit()
            #alg = str(df2['Algorithm'].iloc[0])
            #alg = df2['Algorithm'].iloc[0:1]
            #best_performing[col] = alg
            best_performing_agg = pd.concat([best_performing_agg, best_performing], ignore_index=True)
            #print(best_performing)

    return best_performing_agg
def ploting(dataset, sort_column):
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"

    directory = "Data/eval/5FoldEval/AggregatedEvalYelp/"

    read_file = "AggregatedAll_" + dataset + "_" + sort_column + ".csv"

    df = read_csv(directory + read_file, index_col=False, sep=";")
    print(df)

    # dropping columns if we want to reduce some
    # here we drop precision columns since they do not show well on the picture due to the small values

    df = df.drop(columns=['P_5', 'P_10', 'P_50', 'P_100'])

    # replacing underscore with @ for picture
    df.columns = df.columns.astype(str).str.replace("_", "@")

    #AdaCGL AdaGCL

    df=df.sort_values( 'ndcg@5', ascending=False)
    df_toshow = df[0:15]
    #graphplot = df.plot(kind="bar", x="Algorithm")

    dft = df_toshow.transpose()
    print(dft.columns)
    print(dft)


    #headers = dft.iloc[0]
    #new_df = pd.DataFrame(dft.values[1:], columns=headers)

    dft.rename(columns=dft.iloc[0], inplace=True)
    #dft.rename(columns=df.iloc[0], inplace=True)

    dft.drop(dft.index[0], inplace=True)

    print(dft)
    # selecting only relevant measures

    #pd.Series.str.contains('hit')
    dfsel = dft[dft.index.map(lambda s: s.startswith('RR'))]
    #print(dfsel)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=18)  # controls default text sizes
    #plt.rc('axes', titlesize=15)  # fontsize of the axes title
    #plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    #plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    #plt.rc('legend', fontsize=15)  # legend fontsize
    #plt.rc('figure', titlesize=15)  # fontsize of the figure title

    #graphplot = dft.plot.bar(width=0.97)

    graphplot = dfsel.plot(kind="bar")

    #graphplot = dfsel.plot.bar(width=0.97)

    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.legend(bbox_to_anchor=(0, 0), loc='upper left', borderaxespad=0)
    #plt.legend()
    # legend for the overall figure
    #plt.legend(bbox_to_anchor=(0.9, 1.20), ncol=5)
    # legend for the per metric figure
    plt.legend(bbox_to_anchor=(0.80, 1.3), ncol=3)
    plt.grid(axis='y')

    plt.title("Performance of Selected Recommendation Methods under ReciprocalRank@k on " + dataset + " Dataset")

    plt.xlabel("Evaluation Measure")
    plt.ylabel("Value")

    plt.show()


def getting_subtable(dataset, sort_column):
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"

    directory = "Data/eval/5FoldEval/AggregatedEvalYelp/"

    read_file = "AggregatedAll_" + dataset + "_" + sort_column + ".csv"

    write_file_tex_ndcg = "AggregatedAll_" + dataset + "_bestperforming_ndcg" + ".tex"

    write_file_tex_hit = "AggregatedAll_" + dataset + "_bestperforming_hit" + ".tex"
    write_file_tex_p = "AggregatedAll_" + dataset + "_bestperforming_p" + ".tex"
    write_file_tex_r = "AggregatedAll_" + dataset + "_bestperforming_r" + ".tex"
    write_file_tex_rr = "AggregatedAll_" + dataset + "_bestperforming_rr" + ".tex"

    df = read_csv(directory + read_file, index_col=False, sep=";")
    best = give_best_performing_per_column_asdf(df, 'Yelp')

    print(best)

    best['Measure'] = best['Measure'].str.replace('_', '@')


    best_ndcg = best[best['Measure'].str.contains('ndcg')]

    best_hit = best[best['Measure'].str.contains('hit')]
    best_p = best[best['Measure'].str.contains('P')]

    best_r = best[best['Measure'].str.startswith('R@')]



    best_rr = best[best['Measure'].str.startswith('RR@')]

    s_ndcg = best_ndcg.style \
        .format(precision=5, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1) \
        .hide()

    #s_hit = best_hit.style \
    #    .format(precision=3, thousands=".", decimal=",") \
    #    .format_index(str.upper, axis=1)

    s_hit = best_hit.style \
        .format(precision=5, thousands=".", decimal=",") \
        .hide()

    s_p =best_p.style \
        .format(precision=5, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1) \
        .hide()
    s_r =best_r.style \
        .format(precision=5, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1)\
        .hide()

    s_rr =best_rr.style \
        .format(precision=5, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1)\
        .hide()

    s_ndcg.to_latex(buf=directory + write_file_tex_ndcg, hrules=True, label="table:BestNDCG", caption="Best Performing Methods under NDCG@k for " + dataset + " Dataset")
    s_hit.to_latex(buf=directory + write_file_tex_hit, hrules=True, label="table:BestHIT", caption="Best Performing Methods under HIT@k for " + dataset + " Dataset")
    s_p.to_latex(buf=directory + write_file_tex_p, hrules=True, label="table:BestP", caption="Best Performing Methods under P@k for " + dataset + " Dataset")
    s_r.to_latex(buf=directory + write_file_tex_r, hrules=True, label="table:BestR", caption="Best Performing Methods under Recall@k for " + dataset + " Dataset")
    s_rr.to_latex(buf=directory + write_file_tex_rr, hrules=True, label="table:BestRR", caption="Best Performing Methods under ReciprocalRank@k for " + dataset + " Dataset")
    #print(best_r)
    #best.set_index(['Measure', 'Algorithm'], inplace=True)

    #best.set_index(['Algorithm', 'Measure'], inplace=True)
    #print(best)


    #graphplot = best.plot(kind="bar")

    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.grid(axis='y')


    #plt.title("Performance in HIT of Selected Recommendation Methods on Mindreader Dataset")

    #plt.xlabel("Evaluation Measure")
    #plt.ylabel("Value")

    #plt.show()

def getting_subtable_somecolumns(dataset, sort_column, sort_column_output, columns):
    #directory = "Data/eval/5FoldEval/AggregatedGPUamazonKGCLData/"
    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"

    #directory = "Data/eval/5FoldEval/AggregatedEvalMindreader/"
    #directory = "Data/eval/5FoldEval/AggregatedEvalYelp/"

    read_file = "AggregatedAll_" + dataset + "_" + sort_column + ".csv"

    write_file_tex_baselines = "AggregatedAll_" + dataset + "_bestperforming_baselines_ndcg" + ".tex"

    #write_file_tex_hit = "AggregatedAll_" + dataset + "_bestperforming_hit" + ".tex"
    #write_file_tex_p = "AggregatedAll_" + dataset + "_bestperforming_p" + ".tex"
    #write_file_tex_r = "AggregatedAll_" + dataset + "_bestperforming_r" + ".tex"
    #write_file_tex_rr = "AggregatedAll_" + dataset + "_bestperforming_rr" + ".tex"

    df = read_csv(directory + read_file, index_col=False, sep=";")
    print(df)

    df_select = df[columns]

    df_selectKNN = df_select[df_select['Algorithm'].str.contains('KNN')]

    df_selectAlternate = df_select[df_select['Algorithm'].str.contains('Alternate')]

    df_select = df_select[df_select['Algorithm'].str.endswith('-DP')]

    df_select = pd.concat([df_select, df_selectKNN, df_selectAlternate], ignore_index=True).sort_values(sort_column_output, ascending=False)

    print(df_select)

    #best = give_best_performing_per_column_asdf(df, 'Mindreader')

    #print(best)

    #best['Measure'] = best['Measure'].str.replace('_', '@')


    #best_ndcg = best[best['Measure'].str.contains('ndcg')]

    #best_hit = best[best['Measure'].str.contains('hit')]
    #best_p = best[best['Measure'].str.contains('P')]

    #best_r = best[best['Measure'].str.startswith('R@')]



    #best_rr = best[best['Measure'].str.startswith('RR@')]

    s_baselines = df_select.style \
        .format(precision=5, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1) \
        .hide()

    #s_hit = best_hit.style \
    #    .format(precision=3, thousands=".", decimal=",") \
    #    .format_index(str.upper, axis=1)

    #s_hit = best_hit.style \
    #    .format(precision=5, thousands=".", decimal=",") \
    #    .hide()

    #s_p =best_p.style \
    #    .format(precision=5, thousands=".", decimal=",") \
    #    .format_index(str.upper, axis=1) \
    #    .hide()
    #s_r =best_r.style \
    #    .format(precision=5, thousands=".", decimal=",") \
    #    .format_index(str.upper, axis=1)\
     #   .hide()

    #s_rr =best_rr.style \
    #    .format(precision=5, thousands=".", decimal=",") \
    #    .format_index(str.upper, axis=1)\
    #    .hide()

    s_baselines.to_latex(buf=directory + write_file_tex_baselines, hrules=True, label="table:BestNDCGBaseline" + dataset, caption="Performance of Baseline Methods under NDCG@k for" + dataset + " Dataset")
    #s_hit.to_latex(buf=directory + write_file_tex_hit, hrules=True, label="table:BestHIT", caption="Best Performing Methods under HIT@k for Amazon Book Dataset")
    #s_p.to_latex(buf=directory + write_file_tex_p, hrules=True, label="table:BestP", caption="Best Performing Methods under P@k for Amazon Book Dataset")
    #s_r.to_latex(buf=directory + write_file_tex_r, hrules=True, label="table:BestR", caption="Best Performing Methods under Recall@k for Amazon Book Dataset")
    #s_rr.to_latex(buf=directory + write_file_tex_rr, hrules=True, label="table:BestRR", caption="Best Performing Methods under ReciprocalRank@k for Amazon Book Dataset")
    #print(best_r)
    #best.set_index(['Measure', 'Algorithm'], inplace=True)

    #best.set_index(['Algorithm', 'Measure'], inplace=True)
    #print(best)


    #graphplot = best.plot(kind="bar")

    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.grid(axis='y')


    #plt.title("Performance in HIT of Selected Recommendation Methods on Mindreader Dataset")

    #plt.xlabel("Evaluation Measure")
    #plt.ylabel("Value")

    #plt.show()


def main():
    #AggregatePerEvaluation('KGCL_101', 'amazon')





    #df = CreatedAggregatedResultswithDistinguishingAlgorithm('Yelp', 'ndcg_5')

    #best_performing = give_best_performing_per_column(df, 'Yelp')
    #print(best_performing)


    #format_panda_to_latex_table_fromfile('Yelp', 'ndcg_5')

    ploting('Yelp', 'ndcg_5')
    #getting_subtable_somecolumns('Amazon', 'ndcg_5', 'ndcg_10', ['Algorithm', 'ndcg_10', 'R_10'])
    #getting_subtable('Yelp', 'ndcg_5')
    #remove_agg_label("Data/eval/5FoldEval/AggregatedEvalMindreader/AdditionsofAdaCGLandCGCL/" + "AggregatedAll_Mindreader_ndcg_5.csv")


if __name__ == "__main__":
    main()