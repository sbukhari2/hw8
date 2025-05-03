import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# ========== Section 3.1 – Introducing Pandas Objects ==========

print("========== Section 3.1 ==========\n")

# Load your data
df = pd.read_csv("my_data.csv")

# --- Display the first and last 5 rows ---
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# --- Show .info() and .describe() ---
print("\nDataFrame Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include="all"))

# --- Create a Series with datetime index and slice it ---
date_range = pd.date_range("2025-01-01", periods=12, freq="D")
scores_series = pd.Series(df["Score"].values, index=date_range)

print("\nDatetime Series:")
print(scores_series)

print("\nSliced Series from 2025-01-04 to 2025-01-07:")
print(scores_series["2025-01-04":"2025-01-07"])

# --- Arithmetic on two Series with non-aligned indices ---
s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])

print("\nSeries s1:")
print(s1)

print("\nSeries s2:")
print(s2)

print("\nResult of s1 + s2 (index-aligned):")
print(s1 + s2)

print("\nExplanation: Only indices that match (b and c) are added. Others become NaN.")

# ========== Section 3.2 – Data Indexing and Selection ==========

print("\n========== Section 3.2 ==========\n")

# Create a new custom DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
    "Age": [20, 21, 22, 23, 24],
    "Score": [85, 92, 78, 95, 88]
}
custom_df = pd.DataFrame(data)

# --- .loc vs .iloc ---
print("Using .loc (label-based, includes last index):")
print(custom_df.loc[0:2])  # includes row with label 2

print("\nUsing .iloc (integer-position-based, excludes last):")
print(custom_df.iloc[0:2])  # only rows at positions 0 and 1

# --- Conditional logic to filter rows ---
print("\nRows where Score > 85:")
print(custom_df[custom_df["Score"] > 85])

print("\nRows where Score > 85 and Age < 24:")
print(custom_df[(custom_df["Score"] > 85) & (custom_df["Age"] < 24)])

# --- Boolean mask for values greater than threshold ---
print("\nBoolean mask where Score > 90:")
mask = custom_df["Score"] > 90
print(mask)

print("\nFiltered rows using boolean mask:")
print(custom_df[mask])

# --- Chained indexing pitfall (commented out to avoid warning) ---
print("\nChained indexing (bad practice — NOT executed to avoid warning):")
print("# custom_df[custom_df['Name'] == 'Alice']['Score'] = 100")

# --- Correct usage with .loc (safe assignment) ---
print("\nSafe assignment using .loc:")
custom_df.loc[custom_df["Name"] == "Alice", "Score"] = 100
print(custom_df[custom_df["Name"] == "Alice"])

# --- Set custom index and use .loc/.iloc ---
df2 = custom_df.set_index("Name")

print("\nCustom indexed DataFrame (Name as index):")
print(df2)

print("\nAccess row for 'Bob' using .loc:")
print(df2.loc["Bob"])

print("\nAccess second row using .iloc:")
print(df2.iloc[1])

print("\nColumn slice using .loc (Age to Score):")
print(df2.loc[:, "Age":"Score"])

print("\nColumn slice using .iloc (cols 0 to 1):")
print(df2.iloc[:, 0:2])


print("========== Section 3.3 – Index Objects ==========\n")

# --- Custom index and immutability ---
print("Custom Series with Index:")
s = pd.Series([100, 200, 300], index=["a", "b", "c"])
print(s)

print("\nTrying to modify index label 'a' (will fail):")
try:
    s.index[0] = "x"
except TypeError as e:
    print("Error:", e)

print("\nConclusion: Index objects are immutable.")

# --- Hierarchical index (MultiIndex Series) ---
print("\nMultiIndex Series Example:")
index = pd.MultiIndex.from_tuples([("NY", 2022), ("NY", 2023), ("CA", 2022), ("CA", 2023)],
                                  names=["State", "Year"])
data = pd.Series([500, 550, 450, 470], index=index)
print(data)

print("\nSubset of data for 'NY':")
print(data["NY"])

print("\nSubset for ('CA', 2022):")
print(data[("CA", 2022)])

# --- DataFrame with MultiIndex and .swaplevel(), .sort_index() ---
print("\nMultiIndex DataFrame:")
df = data.unstack()  # Make it a DataFrame
multi_df = df.stack().to_frame(name="Revenue")
multi_df.index.set_names(["Region", "FiscalYear"], inplace=True)
print(multi_df)

print("\nSwap index levels:")
print(multi_df.swaplevel())

print("\nSort by index:")
print(multi_df.sort_index())

# --- Reset and set index, effects of drop and inplace ---
print("\nReset index (default drop=False):")
reset_df = multi_df.reset_index()
print(reset_df)

print("\nSet index back (drop=True):")
set_df = reset_df.set_index(["Region", "FiscalYear"], drop=True)
print(set_df)

# --- Arithmetic with mismatched indices ---
print("\nArithmetic with mismatched indices:")

df1 = pd.DataFrame({
    "A": [1, 2],
    "B": [3, 4]
}, index=["x", "y"])

df2 = pd.DataFrame({
    "B": [10, 20],
    "C": [30, 40]
}, index=["y", "z"])

print("DataFrame 1:")
print(df1)

print("\nDataFrame 2:")
print(df2)

print("\nAddition result (df1 + df2):")
print(df1 + df2)

print("\nExplanation: Addition uses index/column alignment. Missing values = NaN.")



print("========== Section 3.4 – Reindexing ==========\n")

# --- Reindex Series with fill_value and method ---
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print("Original Series:")
print(s)

new_index = ["a", "b", "c", "d", "e"]

print("\nReindexed Series with fill_value=0:")
print(s.reindex(new_index, fill_value=0))

print("\nReindexed with forward fill (method='ffill'):")
print(s.reindex(new_index, method='ffill'))

# --- Reindex DataFrame rows and columns ---
df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
}, index=["x", "y", "z"])

print("\nOriginal DataFrame:")
print(df)

new_rows = ["x", "y", "z", "w"]
new_cols = ["A", "B", "C"]

df_reindexed = df.reindex(index=new_rows, columns=new_cols)

print("\nReindexed DataFrame (with new row and column):")
print(df_reindexed)

print("\nShape:", df_reindexed.shape)
print("Null value count:\n", df_reindexed.isnull().sum())

# --- Simulate time series with missing dates ---
rng = pd.date_range("2025-01-01", periods=5, freq="2D")
ts = pd.Series([100, 120, 130, 150, 160], index=rng)

print("\nOriginal time series (every 2 days):")
print(ts)

full_range = pd.date_range("2025-01-01", "2025-01-09")
ts_reindexed = ts.reindex(full_range)

print("\nReindexed time series (daily, with NaNs):")
print(ts_reindexed)

print("\nInterpolated time series:")
print(ts_reindexed.interpolate())

# --- Reindexing function ---
def smart_reindex(obj, new_index, method=None):
    """
    Reindex a Series or DataFrame to new_index using a method if specified.
    """
    return obj.reindex(new_index, method=method)

print("\nUsing smart_reindex() with forward fill:")
print(smart_reindex(s, ["a", "b", "c", "d"], method="ffill"))

# --- Compare reindex vs sort_index vs sort_values ---
sample = pd.Series([20, 10, 40], index=["b", "a", "c"])
print("\nOriginal sample Series:")
print(sample)

print("\nReindex to ['a', 'b', 'c']:")
print(sample.reindex(["a", "b", "c"]))

print("\nSort by index (sort_index()):")
print(sample.sort_index())

print("\nSort by values (sort_values()):")
print(sample.sort_values())



# ========== Section 3.6 – Operating on Data ==========
print("========== Section 3.6 – Operating on Data ==========\n")

# Sample DataFrame
df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9]
})

# Arithmetic operations & broadcasting
print("Add 10 to all elements:\n", df + 10)
print("\nSubtract row-wise mean (broadcast across columns):\n", df.sub(df.mean(axis=1), axis=0))

# Safe arithmetic with fill values
df2 = pd.DataFrame({
    "A": [1, 2],
    "B": [3, np.nan],
    "D": [5, 6]
})
print("\nUsing .add() with fill_value:\n", df.add(df2, fill_value=0))

# Aggregations
print("\nColumn-wise mean:\n", df.mean())
print("\nRow-wise std dev:\n", df.std(axis=1))

print("\n.agg() across columns:\n", df.agg(['mean', 'min', 'max']))

# Normalize
print("\nNormalize each row (0–1 scale):\n", df.div(df.max(axis=1), axis=0))
print("\nNormalize each column:\n", df.div(df.max(axis=0), axis=1))

# Z-score standardization function
def zscore(df):
    return (df - df.mean()) / df.std()

print("\nZ-score standardized DataFrame:\n", zscore(df))

# ========== Section 3.7 – Handling Missing Data ==========
print("\n========== Section 3.7 – Handling Missing Data ==========\n")

# DataFrame with NaNs
df_nan = pd.DataFrame({
    "A": [1, np.nan, 3],
    "B": [4, 5, np.nan],
    "C": [np.nan, 8, 9]
})
print("Original DataFrame with NaNs:\n", df_nan)

# Locate missing
print("\n.isnull():\n", df_nan.isnull())
print("\n.notnull():\n", df_nan.notnull())

# Drop with threshold
print("\nDrop rows with at least 2 non-NaNs:\n", df_nan.dropna(thresh=2))
print("\nDrop columns with all NaNs:\n", df_nan.dropna(axis=1, how="all"))

# Fill missing
print("\nFill NaNs with 0:\n", df_nan.fillna(0))
print("\nForward fill:\n", df_nan.ffill())
print("\nBackward fill:\n", df_nan.bfill())

# Interpolate
ts = pd.Series([1, np.nan, np.nan, 4], index=pd.date_range("2025-01-01", periods=4))
print("\nTime series before interpolate:\n", ts)
print("\nInterpolated time series:\n", ts.interpolate())

# Updated cleaning pipeline
def clean(df):
    return df.ffill().fillna(0)

print("\nCleaned DataFrame using pipeline:\n", clean(df_nan))

# ========== Section 3.8 – Hierarchical Indexing ==========
print("\n========== Section 3.8 – Hierarchical Indexing ==========\n")

arrays = [
    ["A", "A", "B", "B"],
    [1, 2, 1, 2]
]
index = pd.MultiIndex.from_arrays(arrays, names=["Group", "ID"])
df_mi = pd.DataFrame({"Value": [10, 15, 20, 25]}, index=index)
print("MultiIndex DataFrame:\n", df_mi)

# Tuple-based selection
print("\nSelect (A, 1):\n", df_mi.loc[("A", 1)])

# stack/unstack
print("\nUnstacked:\n", df_mi.unstack())
print("\nRestacked:\n", df_mi.unstack().stack(future_stack=True))

# Cross-section
print("\nCross-section at ID=1:\n", df_mi.xs(1, level="ID"))

# Groupby level aggregation
print("\nGroupby Group, mean:\n", df_mi.groupby(level="Group").mean())

# Rename and flatten
df_mi_flat = df_mi.rename_axis(index=["G", "I"]).reset_index()
print("\nFlattened DataFrame:\n", df_mi_flat)

# ========== Section 3.9 – Concat and Append ==========
print("\n========== Section 3.9 – Concat and Append ==========\n")

# Concatenate Series and DataFrames
s1 = pd.Series([1, 2], index=["a", "b"])
s2 = pd.Series([3, 4], index=["c", "d"])
print("Concatenated Series:\n", pd.concat([s1, s2]))

df1 = pd.DataFrame({"X": [1], "Y": [2]})
df2 = pd.DataFrame({"X": [3], "Y": [4]})
print("\nConcatenated DataFrames:\n", pd.concat([df1, df2]))

# keys and names
print("\nConcat with keys:\n", pd.concat([df1, df2], keys=["first", "second"], names=["Group"]))

# Replace append with concat
print("\nUsing pd.concat() (instead of .append()):\n", pd.concat([df1, df2]))

# Overlapping/non-overlapping
df3 = pd.DataFrame({"X": [5], "Z": [6]})
print("\nConcat with mismatched columns:\n", pd.concat([df1, df3]))

# Combine list of DataFrames
def combine_and_clean(dfs):
    return pd.concat(dfs, ignore_index=True).fillna(0)

combined = combine_and_clean([df1, df3])
print("\nCombined and cleaned DataFrame:\n", combined)

# ========== Section 3.10 – Merging and Joining ==========
print("\n========== Section 3.10 – Merging and Joining ==========\n")

left = pd.DataFrame({
    "ID": [1, 2, 3],
    "Name": ["Alice", "Bob", "Charlie"]
})
right = pd.DataFrame({
    "ID": [2, 3, 4],
    "Score": [90, 85, 95]
})

# One-to-one merge
print("Merge on ID:\n", pd.merge(left, right, on="ID"))

# Many-to-one merge
dept = pd.DataFrame({
    "DeptID": [1, 2],
    "DeptName": ["HR", "Finance"]
})
employee = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "DeptID": [1, 2]
})
print("\nMany-to-one merge:\n", pd.merge(employee, dept, on="DeptID"))

# Many-to-many
courses = pd.DataFrame({
    "Student": ["Alice", "Alice", "Bob"],
    "Course": ["Math", "Science", "Math"]
})
grades = pd.DataFrame({
    "Course": ["Math", "Science"],
    "Credit": [3, 4]
})
print("\nMany-to-many merge:\n", pd.merge(courses, grades, on="Course"))

# Join on index
df_left = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
df_right = pd.DataFrame({"b": [3, 4]}, index=["y", "z"])
print("\nJoin on index:\n", df_left.join(df_right, how="outer"))

# Merge 3+ with indicator
m1 = pd.DataFrame({"ID": [1, 2]})
m2 = pd.DataFrame({"ID": [2, 3]})
m3 = pd.DataFrame({"ID": [3, 4]})

merged = pd.merge(m1, m2, on="ID", how="outer", indicator="Merge_1_2")
merged = pd.merge(merged, m3, on="ID", how="outer", indicator="Final_Merge")
print("\nMerge 3 DataFrames with indicators:\n", merged)



# Load CSVs
area_df = pd.read_csv("state_area.csv")
abbr_df = pd.read_csv("state_abbr.csv")
pop_df = pd.read_csv("state_population.csv")

# Merge all three on 'State'
merged_df = pd.merge(area_df, abbr_df, on="State")
merged_df = pd.merge(merged_df, pop_df, on="State")

print("Merged DataFrame:")
print(merged_df)

# Compute population density
merged_df["Density"] = merged_df["Population"] / merged_df["Area"]
print("\nDataFrame with Population Density:")
print(merged_df)

# Filter top 10 most densely populated states (we only have 5 here)
top_density = merged_df.sort_values(by="Density", ascending=False).head(10)
print("\nTop Densely Populated States:")
print(top_density[["State", "Density"]])

# Add population size category
def categorize_population(pop):
    if pop > 20000000:
        return "large"
    elif pop > 10000000:
        return "medium"
    else:
        return "small"

merged_df["PopSize"] = merged_df["Population"].apply(categorize_population)
print("\nWith Population Size Category:")
print(merged_df[["State", "Population", "PopSize"]])

# Scatter plot of Population vs. Area
plt.figure(figsize=(10, 6))
colors = merged_df["Population"].apply(lambda x: "red" if x > 10000000 else "blue")

plt.scatter(merged_df["Area"], merged_df["Population"], c=colors)
for i, row in merged_df.iterrows():
    if row["Population"] > 10000000:
        plt.text(row["Area"], row["Population"], row["Abbreviation"], fontsize=9, ha='right')

plt.xlabel("Area (sq km)")
plt.ylabel("Population")
plt.title("US State Population vs Area")
plt.grid(True)
plt.tight_layout()
plt.show()
