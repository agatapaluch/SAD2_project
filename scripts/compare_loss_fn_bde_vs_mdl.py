import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def rank_biserial_correlation(x, y):
    """
    Effect size for Mann-Whitney U.
    Range: [-1, 1]
    """
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    n1 = len(x)
    n2 = len(y)
    return 1.0 - (2.0 * u) / (n1 * n2)


def compare_medians(df1, df2, column, alpha=0.05):
    x = df1[column].dropna().values
    y = df2[column].dropna().values

    med_x = np.median(x)
    med_y = np.median(y)

    u, p = mannwhitneyu(x, y, alternative="two-sided")
    rbc = rank_biserial_correlation(x, y)

    print "\n=== %s ===" % column
    print "n1 =", len(x), ", n2 =", len(y)
    print "Median (BDe) =", round(med_x, 4)
    print "Median (MDL) =", round(med_y, 4)
    print "Median difference =", round(med_x - med_y, 4)
    print "Mann-Whitney U =", round(u, 4)
    print "p-value =", p
    print "Rank-biserial correlation =", round(rbc, 4)

    if p < alpha:
        print "Result: SIGNIFICANT difference in medians"
    else:
        print "Result: no significant difference in medians"


def main():
    columns = [
        "edge_jaccard_distance",
        "graph_edit_distance"
    ]

    df_bde = pd.read_csv("results_final_BDE/group_eval.csv")
    df_mdl = pd.read_csv("results_final_MDL/group_eval.csv")

    for col in columns:
        compare_medians(df_bde, df_mdl, col)


if __name__ == "__main__":
    main()        