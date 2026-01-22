import pandas as pd
from scipy.stats import shapiro


def test_normality(df, df_name, columns, alpha=0.05):
    print("Normality tests for {}".format(df_name))

    for col in columns:
        data = df[col].dropna()

        print("\nColumn: {}".format(col))
        print("Sample size: n = {}".format(len(data)))

        if len(data) < 3:
            print("Not enough data to test normality (n < 3)")
            continue

        stat, p = shapiro(data)

        print("Shapiro-Wilk statistic = {}".format(stat))
        print("p-value = {}".format(p))

        if p > alpha:
            print("Result: Consistent with normality")
        else:
            print("Result: Deviates from normality")


def main():
    columns = [
        "edge_jaccard_distance",
        "graph_edit_distance"
    ]

    df_bde = pd.read_csv("results_final_BDE/group_eval.csv")
    df_mdl = pd.read_csv("results_final_MDL/group_eval.csv")

    test_normality(df_bde, "BDE", columns)
    test_normality(df_mdl, "MDL", columns)


if __name__ == "__main__":
    main()