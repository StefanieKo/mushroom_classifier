# ===========================================================
# Mushroom Classification with Decision Trees
# Comparative Tree Analysis: Model Performance, Feature Importance, and Tree Visualization
# ===========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# ===========================================================
# 1. Load Data
# ===========================================================
url = "https://raw.githubusercontent.com/cornelius31415/DATA-SCIENCE/main/Mushroom%20Classification/mushroom.csv"
mushroom_data = pd.read_csv(url)

# ===========================================================
# 2. Quick Overview of the Dataset
# ===========================================================
print("\n--- Shape (rows, columns) ---")
print(mushroom_data.shape)

print("\n--- Column Names ---")
print(mushroom_data.columns.tolist())

print("\n--- First 5 Rows ---")
print(mushroom_data.head())

print("\n--- Data Types ---")
print(mushroom_data.info())

print("\n--- Target Distribution ---")
print(mushroom_data['class'].value_counts())

print("\n--- Unique Values per Feature ---")
print(mushroom_data.nunique())

# ===========================================================
# 3. Feature and Target Preparation (with One-Hot-Encoding)
# ===========================================================
numeric_features = ["cap-diameter", "stem-height", "stem-width"]
categorical_features = ["cap-shape", "gill-attachment", "gill-color", "stem-color", "season"]

# One-Hot-Encoding for categorical features
X = pd.get_dummies(
    mushroom_data.drop("class", axis=1),
    columns=categorical_features,
    drop_first=False
)
y = mushroom_data["class"]  # already 0/1 encoded

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===========================================================
# 4. Train Decision Trees with Various Depths
# ===========================================================
depths = [None] + list(range(1, 21))
results = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results.append((d, acc, prec, rec, tree))


# ===========================================================
# 5. Performance DataFrame and Plot
# ===========================================================
results_df = pd.DataFrame(results, columns=["max_depth", "Accuracy", "Precision", "Recall", "Model"])
print("\n--- Model Performance ---")
print(results_df.drop(columns="Model"))

plt.figure(figsize=(8,5))
x_vals = [0 if r[0] is None else r[0] for r in results]  # None -> 0
plt.plot(x_vals, [r[1] for r in results], marker='o', label="Accuracy")
plt.plot(x_vals, [r[2] for r in results], marker='s', label="Precision")
plt.plot(x_vals, [r[3] for r in results], marker='^', label="Recall")
plt.xticks(x_vals)
plt.xlabel("Tree Depth")
plt.ylabel("Score")
plt.title("Performance vs Tree Depth")
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================
# 6. Aggregated Feature Importances
# ===========================================================
selected_depths = [None, 1, 2, 3, 4, 5, 10, 20]
all_importances = pd.DataFrame(index=X_train.columns)

for d, acc, prec, rec, tree in results:
    if d in selected_depths:
        imp = pd.Series(tree.feature_importances_, index=X_train.columns)
        all_importances[f"depth={d}"] = imp

all_importances = all_importances.fillna(0)

# Group dummy variables back to original features
groups = {
    "Cap Diameter": ["cap-diameter"],
    "Stem Height": ["stem-height"],
    "Stem Width": ["stem-width"],
    "Cap Shape": [c for c in X_train.columns if c.startswith("cap-shape")],
    "Gill Attachment": [c for c in X_train.columns if c.startswith("gill-attachment")],
    "Gill Color": [c for c in X_train.columns if c.startswith("gill-color")],
    "Stem Color": [c for c in X_train.columns if c.startswith("stem-color")],
    "Season": [c for c in X_train.columns if c.startswith("season")]
}

def aggregate_importances(importances, groups):
    agg = pd.DataFrame(index=groups.keys())
    for col in importances.columns:
        vals = {}
        for group, feats in groups.items():
            vals[group] = importances.loc[importances.index.isin(feats), col].sum()
        agg[col] = pd.Series(vals)
    return agg

agg_importances = aggregate_importances(all_importances, groups)

# Barplot
agg_importances.T.plot(kind="bar", figsize=(10,6))
plt.ylabel("Aggregated Feature Importance")
plt.title("Feature Importance by Variable Group (One-Hot Encoded)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''

# ===========================================================
# 7. Plot Example Decision Trees
# ===========================================================
for d in [1, 2, 3, 4, 5]:
    tree_model = DecisionTreeClassifier(max_depth=d, random_state=1)
    tree_model.fit(X_train, y_train)

    plt.figure(figsize=(15,8))
    plot_tree(
        tree_model,
        feature_names=X_train.columns,
        class_names=["Edible", "Poisonous"],
        filled=True,
        fontsize=8
    )
    plt.title(f"Decision Tree (max_depth={d})")
    plt.show()
    plt.savefig("tree.svg")
'''

for d in [1, 2, 3, 4, 5]:
    tree_model = DecisionTreeClassifier(max_depth=d, random_state=1)
    tree_model.fit(X_train, y_train)

    plt.figure(figsize=(15,8))
    plot_tree(
        tree_model,
        feature_names=X_train.columns,
        class_names=["Edible", "Poisonous"],
        filled=True,
        fontsize=8
    )
    filename = f"tree_{d}.svg"  # eindeutiger Name je Baum
    plt.savefig(filename, format="svg")
    plt.close()
