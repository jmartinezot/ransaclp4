'''
This module contains functions for statistical analysis.
'''
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, chi2, norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from math import factorial

# Create a custom function to convert the DataFrame to good data for a heatmap
# the values < 0.05 and "good" for the algorithm in the column are going to be 2 - value
# the values < 0.05 and "bad" for the algorithm in the row are going to be value
# the rest of values are going to be 1
# this function returns a DataFrame with the same shape as the original one
# always keep track of the name of the row and column of each cell,
# because the heatmap will be created using the row and column names
# a value is good if means[row_name] > means[col_name]
# a value is bad if means[row_name] < means[col_name]
def to_heatmap_custom(df, means):
    heatmap_data = pd.DataFrame(1, index=df.index, columns=df.columns)
    for i, row_name in enumerate(df.index):
        for j, col_name in enumerate(df.columns):
            value = df.loc[row_name, col_name]
            if value < 0.05:
                heatmap_data.loc[row_name, col_name] = 2 - value if means[i] > means[j] else value 
            if i == j:
                heatmap_data.loc[row_name, col_name] = 3      
    return heatmap_data

def to_image_custom(df_original, df_heatmap_data, filename, df_p_values = None):
    print("Entering to_image_custom")
    # Create a colormap from the list of colors
    cmap = ListedColormap(['red', 'gray', 'green', 'white'])
    sns.set(font_scale=1.2)
    plt.figure(figsize=(12, 10))
    col_labels = df_original.columns
    row_labels = df_original.index
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
    # Create a copy of the p-values DataFrame to format the non-empty values
    p_values_formatted = df_p_values.copy()
    # Iterate over rows and columns to format non-empty values
    for i, row_name in enumerate(df_p_values.index):
        for j, col_name in enumerate(df_p_values.columns):
            val = df_p_values.loc[row_name, col_name]
            if val != "":
                p_values_formatted.loc[row_name, col_name] = f"{float(val):.2f}"

    if df_p_values is not None:
        sns.heatmap(df_heatmap_data, fmt="", linewidths=.5, cbar=False, cmap=cmap, xticklabels=col_labels, yticklabels=row_labels, annot=p_values_formatted.values)  # Use df_p_values as annotations
    else:
        sns.heatmap(df_heatmap_data, fmt="d", linewidths=.5, cbar=False, cmap=cmap, xticklabels=col_labels, yticklabels=row_labels)
    # save the figure
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()

def rankMatrix(data, decreasing=True):
    def rank_rows(row):
        if decreasing:
            return rankdata(-row, method='average')
        else:
            return rankdata(row, method='average')
    
    ranked_data = data.apply(rank_rows, axis=1)
    return ranked_data

def friedmanTest(data):
    N, k = data.shape
    mr = rankMatrix(data).mean(axis=0)
    
    friedman_stat = 12 * N / (k * (k + 1)) * (np.sum(mr**2) - k * (k + 1)**2 / 4)
    p_value = 1 - chi2.cdf(friedman_stat, df=k - 1)

    result = {
        "Friedman's chi-squared": friedman_stat,
        "df": k - 1,
        "p-value": p_value,
        "method": "Friedman's rank sum test",
        "data_name": str(data.columns.values)
    }
    return result

def processControlColumn(data, control):
    if control is not None:
        if isinstance(control, str):
            if control in data.columns:
                control = data.columns.get_loc(control)
            else:
                raise ValueError("The name of the column to be used as control does not exist in the data matrix")
        elif control > data.shape[1] - 1 or control < 0:
            raise ValueError(f"Non-valid value for the control parameter. It has to be either the name of a column or a number between 0 and {data.shape[1] - 1}")
    return control

def generatePairs(k, control):
    if control is None:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        pairs = [(control, i) for i in range(k) if i != control]
    return pairs

def buildPvalMatrix(pvalues, k, pairs, cnames, control):
    if control is None:
        matrix_raw = pd.DataFrame(np.nan, index=cnames, columns=cnames)
        for (i, j), pval in zip(pairs, pvalues):
            matrix_raw.iloc[i, j] = pval
            matrix_raw.iloc[j, i] = pval
    else:
        matrix_raw = pd.DataFrame(np.nan, columns=cnames, index=[0])
        for _, j in pairs:
            matrix_raw.iloc[0, j] = pvalues[j]
    return matrix_raw

def correctForMonotocity(pvalues):
    for i in range(1, len(pvalues)):
        pvalues[i] = max(pvalues[i], pvalues[i - 1])
    return pvalues

def friedmanPost(data, control=None):
    k = data.shape[1]
    N = data.shape[0]
    control = processControlColumn(data, control)
    pairs = generatePairs(k, control)
    
    mean_rank = rankdata(-data.values, axis=1, method='average').mean(axis=0)
    sd = np.sqrt(k * (k + 1) / (6 * N))
    pvalues = [2 * (1 - norm.cdf(abs(mean_rank[i] - mean_rank[j]) / sd)) for i, j in pairs]

    matrix_raw = buildPvalMatrix(pvalues, k, pairs, data.columns, control)
    return matrix_raw

def setUpperBound(vector, bound):
    return np.minimum(vector, bound)

def countRecursively(k):
    res = [0]
    if k > 1:
        res += countRecursively(k - 1)
        for j in range(2, k + 1):
            additional_values = [int((x + factorial(j) / (2 * factorial(j - 2)))) for x in countRecursively(k - j)]
            res += additional_values
    return sorted(set(res))

def adjustShaffer(raw_matrix):
    if not isinstance(raw_matrix, (pd.DataFrame, np.ndarray)):
        raise ValueError("This correction method requires a DataFrame or ndarray with the p-values of all the pairwise comparisons.")
    
    if raw_matrix.shape[0] != raw_matrix.shape[1]:
        raise ValueError("This correction method requires a square matrix or DataFrame with the p-values of all the pairwise comparisons.")
    
    k = raw_matrix.shape[0]
    pairs = generatePairs(k, None)
    raw_pvalues = [raw_matrix.iloc[i, j] for i, j in pairs]

    sk = countRecursively(k)[1:]  # Exclude 0 from the count

    # Replicate elements of sk (excluding the last one) according to the differences between elements
    replicated = np.repeat(sk[:-1], np.diff(sk))

    # Get the last element of sk
    last_element = sk[-1]

    # Combine the replicated part with the last element
    t_i = np.concatenate([replicated, [last_element]])
    t_i = list(t_i)
    t_i.reverse()

    o = np.argsort(raw_pvalues)
    adj_pvalues = np.array(raw_pvalues)[o] * t_i
    adj_pvalues = setUpperBound(adj_pvalues, 1) 
    adj_pvalues = correctForMonotocity(adj_pvalues)

    # Restoring original order
    adj_pvalues = adj_pvalues[o.argsort()]

    # Regenerating the matrix
    adj_matrix = raw_matrix.copy()
    for (i, j), pval in zip(pairs, adj_pvalues):
        adj_matrix.iloc[i, j] = adj_matrix.iloc[j, i] = pval

    return adj_matrix

def determine_color(val, row_name, col_name, means):
    if val < 0.05:
        if means[row_name] > means[col_name]:
            return 'green'
        else:
            return 'red'
    else:
        return 'gray'
    
def format_text(val):
    return f"{val:.2f}"







