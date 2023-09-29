import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


OUT = './output1'
RUN_TSNE = False
RUN_DENDROGRAM = False
START = datetime.now()

# Create output directory
os.makedirs(OUT, exist_ok=True)

# Load the data
medical_clean = pd.read_csv('./medical_clean.csv')

# View the data
print(f'\n{medical_clean.head().to_string()}\n')

# Evaluate the data structures
print(f'\ncontinuous data:\n{medical_clean.describe().T.to_string()}')
cat_data = medical_clean[[col for col in medical_clean.columns
                          if medical_clean[col].dtype not in ['float64',
                                                              'int64']]]
print(f'\ncategorical data:\n{cat_data.describe().T.to_string()}\n')

# Evaluate data types and check for nulls
medical_clean.info()

# Check for duplicates
print(f'\nduplicates:\n{medical_clean.duplicated().sum()}')

# Check for outliers
num_cols = [col for col in medical_clean.columns
            if medical_clean[col].dtype in ['float64', 'int64']]
df_num = medical_clean[num_cols]
df_zscores = df_num.apply(stats.zscore)
df_outliers = df_zscores.apply(lambda x: (x > 3) | (x < -3))
print(f'\noutliers:\n{df_outliers.sum().to_string()}')

# Subset features of interest
features = ['Initial_admin', 'HighBlood', 'Stroke', 'Complication_risk',
            'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia',
            'BackPain', 'Anxiety', 'Allergic_rhinitis',
            'Reflux_esophagitis', 'Asthma', 'Services', 'Initial_days',
            'TotalCharge']
df = medical_clean[features]

# Normalize numeric data
num_cols = [col for col in features
            if df[col].dtype in ['float64', 'int64']]
scaler = StandardScaler()
df.loc[:, num_cols] = scaler.fit_transform(df[num_cols])
print(df[num_cols].head().to_string())

# Encode categorical features
df.replace({'Yes': 1, 'No': 0}, inplace=True)
df = pd.get_dummies(df)
df.drop(['Initial_admin_Elective Admission',
         'Initial_admin_Observation Admission',
         'Complication_risk_High', 'Services_Blood Work',
         'Services_Intravenous'],
        axis=1, inplace=True)
df.columns = [col.replace(' ', '_') for col in df.columns]
print(df.head().to_string())

# Save the cleaned dataset
df.to_csv(os.path.join(OUT, 'cleaned_dataset.csv'))

# View flattened clusters
if RUN_TSNE:
    plt.subplots()
    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(df)
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    sns.scatterplot(x=xs, y=ys)
    plt.title('t-SNE plot')
    plt.savefig(os.path.join(OUT, 't-sne_plot.jpg'))
    plt.show()

# Perform hierarchical linkage and view dendrogram
mergings = linkage(df.values, method='ward')
if RUN_DENDROGRAM:
    plt.subplots()
    plt.subplots(figsize=(10, 6))
    dendrogram(mergings)
    plt.title('Ward Linkage')
    plt.savefig(os.path.join(OUT, 'ward_linkage.jpg'))
    plt.show()

# Label the clusters and view them
df['cluster_labels'] = fcluster(mergings, 2, criterion='maxclust')
if RUN_TSNE:
    plt.subplots()
    sns.scatterplot(x=xs, y=ys, hue=df.cluster_labels)
    plt.title('Labeled t-SNE plot')
    plt.savefig(os.path.join(OUT, 'labeled_t-sne_plot.jpg'))
    plt.show()

# Assess accuracy
score = silhouette_score(df.loc[:, df.columns.__ne__('cluster_labels')],
                         df.cluster_labels)
print('Silhouette Score:', score, '\n')

# Summarize each of the clusters
for f in features:
    if f not in ['TotalCharge', 'Initial_days']:
        ct = pd.crosstab(df.cluster_labels, medical_clean[f])
        print(ct.to_string(), '\n')
summary = pd.concat([df.cluster_labels,
                     medical_clean[['TotalCharge', 'Initial_days']]],
                    axis=1).groupby('cluster_labels').mean()
print(summary.to_string(), '\n')

# Calculate the readmission rates
rr_df = pd.concat([df.cluster_labels, medical_clean.ReAdmis], axis=1)
rr_df.replace({'Yes': 1, 'No': 0}, inplace=True)
readmission_rate = rr_df.groupby('cluster_labels').mean()
print(readmission_rate.to_string(), '\n')

print('Start Time:', START, '\n',
      'End Time:', datetime.now(), '\n',
      'Elapsed Time:', datetime.now() - START)
