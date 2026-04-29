#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:39:28 2026

@author: sampadasaxena
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
red = pd.read_csv("winequality-red.csv", sep=";")

# -----------------------------
# STEP 1: Basic Info
# -----------------------------
print(red.info())
print(red.describe())

# -----------------------------
# STEP 2: Visualization BEFORE
# -----------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=red)
plt.xticks(rotation=90)
plt.title("Boxplot Before Outlier Removal")
plt.show()

red.hist(figsize=(12,10))
plt.show()

plt.scatter(red['alcohol'], red['quality'])
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Alcohol vs Quality (Before)")
plt.show()

# -----------------------------
# STEP 3: Separate features & target
# -----------------------------
features = red.drop('quality', axis=1)

# -----------------------------
# STEP 4: Z-Score
# -----------------------------
from scipy import stats

z = np.abs(stats.zscore(features))
red_zscore = red[(z < 3).all(axis=1)]

print("Original shape:", red.shape)
print("After Z-score:", red_zscore.shape)

# -----------------------------
# STEP 5: IQR
# -----------------------------
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)

IQR = Q3 - Q1

red_iqr = red[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]

print("After IQR:", red_iqr.shape)

# -----------------------------
# STEP 6: Isolation Forest
# -----------------------------
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
pred = iso.fit_predict(features)

red_iso = red[pred == 1]

print("After Isolation Forest:", red_iso.shape)

# -----------------------------
# STEP 7: Rows removed
# -----------------------------
print("Rows removed (Z-score):", red.shape[0] - red_zscore.shape[0])
print("Rows removed (IQR):", red.shape[0] - red_iqr.shape[0])
print("Rows removed (ISO):", red.shape[0] - red_iso.shape[0])

# -----------------------------
# STEP 8: Visualization AFTER
# -----------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=red_iqr)
plt.xticks(rotation=90)
plt.title("Boxplot After IQR")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=red_zscore)
plt.xticks(rotation=90)
plt.title("Boxplot After Z-score")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=red_iso)
plt.xticks(rotation=90)
plt.title("Boxplot After Isolation Forest")
plt.show()

