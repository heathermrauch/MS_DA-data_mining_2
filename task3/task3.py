import os

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

OUT = './output3'

# Create output directory
os.makedirs(OUT, exist_ok=True)

# Load and view the data
df = pd.read_csv('medical_market_basket.csv')
print('Data Preview:\n', df.head().to_string(), '\n')
df.info()
print()

# Check for entirely missing rows
print('Rows missing all fields:', df.loc[df.isnull().all(axis=1)].shape[0])
df = df.loc[~df.isnull().all(axis=1)]
print(f'Remaining rows:', df.shape[0], '\n')

# Convert missing values to empty strings
df.replace(np.nan, '', inplace=True)

# Convert DataFrame into list of item sets
patients = df.apply(set, axis=1).to_list()
print('Converted (top 5):')
for i in range(5):
    print(i, patients[i])
print()

# Export unique prescriptions for aggregation
prescriptions = set(rx for patient in patients for rx in patient)
prescriptions.remove('')
print('Unique Prescriptions:', len(prescriptions), '\n')
pd.DataFrame(prescriptions, columns=['prescriptions']).to_csv(os.path.join(OUT, 'prescriptions.csv'), index=False)

""" 
    Manual mapping performed using https://drugs.com
    prescriptions -> drug_class -> group
    References listed in prescription_map.csv
"""

# Read in prescription map
rx_map_path = 'prescription_map.csv'
rx_map_df = pd.read_csv(rx_map_path)
rx_map = dict(zip(rx_map_df.prescriptions, rx_map_df.group))

# Aggregate prescriptions using rx_map
patients = [list(set(rx_map[rx] for rx in patient if rx != '')) for patient in patients]
print('Aggregated (top 5):')
for i in range(5):
    print(i, patients[i])
print()

# One-hot encode item sets
encoder = TransactionEncoder().fit(patients)
data = pd.DataFrame(encoder.transform(patients), columns=encoder.columns_)
print('Encoded:\n', data.head(), '\n')

# Save cleaned data
data.to_csv(os.path.join(OUT, 'cleaned_data.csv'), index=False)

# Apply Apriori Algorithm
frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
print('Number of frequent itemsets:', len(frequent_itemsets))
print('Top 10 itemsets:')
print(frequent_itemsets. sort_values('support', ascending=False). head(10).to_string(), '\n')

# Compute association rules
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
print('Number of rules:', rules.shape[0])
print('Top rules:')
print(rules[['antecedents', 'consequents']].head().to_string(), '\n')

# Scatterplot of all rules by support and lift
fig, _ = plt.subplots()
fig.suptitle('Prescription Association Rules\nby Support, Lift, and Confidence')
sns.scatterplot(x='antecedent support', y='consequent support',
                size='lift', hue='confidence', data=rules, palette='crest')
plt.axhline(y=0.01, alpha=0.1)
plt.axvline(x=0.01, alpha=0.1)
plt.savefig(os.path.join(OUT, 'rules_scatterplot.png'))
plt.show()

# Filter association rules
filtered_rules = rules.loc[rules.confidence.ge(0.6)]
print('Number of filtered rules:', filtered_rules.shape[0])
print('Filtered rules:')
print(filtered_rules[['antecedents', 'consequents', 'support', 'lift', 'confidence']].to_string(), '\n')

# Heatmap of filtered association rules
plot_df = filtered_rules.copy()
plot_df.loc[:, 'antecedents'] = plot_df.loc[:, 'antecedents'].apply(lambda x: ','.join(list(x)))
plot_df.loc[:, 'consequents'] = plot_df.loc[:, 'consequents'].apply(lambda x: ','.join(list(x)))
pivot = plot_df.pivot(index='antecedents', columns='consequents', values='support')
fig, _ = plt.subplots()
fig.set_tight_layout(True)
fig.suptitle('Filtered Association Rules by Support')
sns.heatmap(pivot.sort_values(pivot.columns[0], ascending=False), annot=True, cbar=False, cmap='crest')
plt.savefig(os.path.join(OUT, 'filtered_rules_heatmap.png'))
plt.show()

# Top 3 rule summaries
top_three = filtered_rules.sort_values('support', ascending=False).iloc[:3]
print('Top three association rule summaries')
for i in range(3):
    print(top_three.iloc[i])
