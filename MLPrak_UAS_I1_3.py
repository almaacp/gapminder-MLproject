#%% import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fcmeans import FCM
import warnings
warnings.filterwarnings("ignore")

#%% import data
df = pd.read_csv('gapminder_full.csv')
print(df.head())

#%% amati bentuk data
df.shape

#%% Melihat ringkasan statistik deskriptif dari DataFrame 
print(df.describe())

#%% cek null data
print(df.isnull().sum())

#%% cek outlier
Q1 = df.select_dtypes(include=np.number).quantile(0.25)
Q3 = df.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

outliers = ((df.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | 
            (df.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)
print(df[outliers])

numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 9))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df[col]) # type: ignore
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

#%% menghilangkan outlier (jalankan bagian ini secara berulang hingga tidak ada outlier)
lower_bound = Q1[df.select_dtypes(include=np.number).columns] - 1.5 * IQR
upper_bound = Q3[df.select_dtypes(include=np.number).columns] + 1.5 * IQR
outlier_filter = ((df.select_dtypes(include=np.number) < lower_bound) | (df.select_dtypes(include=np.number) > upper_bound)).any(axis=1)

df = df[~outlier_filter]

Q1 = df.select_dtypes(include=np.number).quantile(0.25)
Q3 = df.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

outliers = ((df.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | (df.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)
print(df[outliers])

#%% Boxplot setelah outlier dihapus
numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 9))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df[col]) # type: ignore
    plt.title(f'Boxplot for {col}')
    plt.tight_layout()
plt.show()

df.shape

# %% amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['year','life_exp','population','gdp_cap']:
  n += 1
  plt.subplot(2,2,n)
  plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
  sns.histplot(
    df[x], kde=True, # type: ignore
    stat="density", kde_kws=dict(cut=3), bins = 20)
  plt.title('Distplot of {}'.format(x))
plt.show()

#%% Ploting untuk mencari relasi antara year, life_exp, population, dan gdp_cap
plt.figure(1 , figsize = (15 , 20))
n = 0
for x in ['year','life_exp','population','gdp_cap']:
  for y in ['year','life_exp','population','gdp_cap']:
    n += 1
    plt.subplot(4,4,n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.regplot(x = x , y = y , data = df)
    plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()

#%% Melihat sebaran life_exp dan gdp_cap pada continent
df.groupby('continent').size()
plt.figure(1 , figsize = (15 , 7))
for continent in ['Africa','Americas','Asia','Europe','Oceania']:
  plt.scatter(x = 'life_exp',y = 'gdp_cap' ,
  data = df[df['continent'] == continent] ,s = 200 , alpha = 0.5,
  label = continent)
  plt.xlabel('Life Expectancy (years)'), plt.ylabel('GDP per Capita (US$)') # type: ignore
  plt.legend()
plt.show()

#%% Merancang K-Means untuk life_exp vs gdp_cap
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[['life_exp','gdp_cap']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
  algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10,
                      max_iter=300, random_state= 111) )
  algorithm.fit(X1)
  inertia.append(algorithm.inertia_)

# Plot bentuk visual elbow
plt.figure(1 , figsize = (15 ,6))
plt.plot(range(1 , 11) , inertia , 'o')
plt.plot(range(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia') # type: ignore

#%% Membangun K-Means
algorithm = (KMeans(n_clusters = 3,init='k-means++', n_init = 10, 
                    max_iter=300, tol=0.0001, random_state= 111 , algorithm='elkan') )
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

#%% Menyiapkan data untuk bentuk visual cluster
step = 0.5
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # array diratakan 1D

#%% Melihat bentuk visual cluster
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1 , interpolation='nearest',
extent=(xx.min(), xx.max(), yy.min(), yy.max()),
cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower') # type: ignore
plt.scatter( x = 'life_exp' ,y = 'gdp_cap' , data= df , c = labels2 , s = 75 )
plt.scatter(x = centroids2[: , 0] , y = centroids2[: , 1] , s = 300 , 
            c = 'red' , alpha = 0.5)
plt.ylabel('GDP per Capita (US$)'),plt.xlabel('Life Expectancy (years)') # type: ignore
plt.show()

#%% Melihat nilai Silhouette Score
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)

#%% Membangun FCM
nmpy = df.drop(columns=['country','continent','year','population']).values
model = FCM(n_clusters=3) # type: ignore
model.fit(nmpy)
centers = model.centers
labels = model.predict(nmpy)
plt.scatter(nmpy[labels == 0, 0], nmpy[labels == 0, 1], s=30, c='r', alpha=0.4)
plt.scatter(nmpy[labels == 1, 0], nmpy[labels == 1, 1], s=30, c='b', alpha=0.4)
plt.scatter(nmpy[labels == 2, 0], nmpy[labels == 2, 1], s=30, c='g', alpha=0.4)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='black', marker='+')
plt.title('Clustering')
plt.show()

