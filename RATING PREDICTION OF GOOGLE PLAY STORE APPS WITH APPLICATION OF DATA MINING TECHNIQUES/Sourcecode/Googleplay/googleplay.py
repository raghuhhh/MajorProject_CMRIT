#Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import cycle, islice
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('googleplaystore.csv')
print(df.info())
print(df['Category'].value_counts())

#Data Visualization

'''total no of Apps per Category'''
plt.figure()
my_colors = list(islice(cycle(['b', 'r', 'g', 'c', 'y', 'm','k']), None, len(df))) # a way to represent different colours 

df.Category.value_counts().plot(kind = 'bar', color = my_colors, figsize = (10,8), title = 'Number of apps in each category');
plt.xlabel('Category')
plt.ylabel('Frequency');

# Piechart
plt.figure()
df['Category'].value_counts().plot.pie(y = df['Category'], figsize = (15, 16), label = '', autopct = '%1.1f%%', title = 'Distribution of apps by category', );

'''Ratings'''
plt.figure()
# We clean the Rating column with the help of lambda function
df['Rating'] = df['Rating'].apply(lambda x: str(x).replace('nan', 'NaN') if 'nan' in str(x) else str(x))  # Replace nan with NaN
df['Rating'] = df['Rating'].apply(lambda x: float(x)) # Rewrite column in a float format

plt.figure()
df['Rating'] = df['Rating'].fillna(df['Rating'].median()) # Replace null numbers with median numbers

df.Rating.isnull().sum()
plt.figure()
df.Rating.value_counts().plot(kind = 'bar', stacked = True, figsize = (12, 8), title = 'Distribution of Rating among apps available on Google Play Store'); # Historgram of frequencies 
plt.xlabel('Rating')
plt.ylabel('Frequencies');

#Lets remove the comma & the plus sign
plt.figure()
df2=df.copy()
df2["Installs"]=df["Installs"].str.replace(",","").str[:-1]
df2["Installs"]=pd.to_numeric(df2["Installs"],errors='coerce')
ginstalls=df2.groupby("Category").sum()["Installs"].sort_values().head(10)
ginstalls.plot(kind="barh")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.title("Installs by category")

#Analysing the family category
plt.figure()
family_category=df2[df2["Category"]=="FAMILY"]
fam=family_category.groupby("Genres").size().sort_values(ascending=False).head(5)
fam.plot(kind="bar",color="lightblue")
plt.title("Family Category (Number of apps)")
plt.ylabel("Number of Apps")

plt.figure()
genres=["Education","Simulation","Entertainment","Puzzle","Casual"]
fam2=family_category[family_category["Genres"].isin(genres)]
groupfam2=fam2.groupby("Genres").sum()["Installs"]
groupfam2.plot(kind="bar",color="pink")

#Ratings & Categories
plt.figure()
Catratings=df.groupby("Category").mean().drop("1.9").head(10)
print(Catratings)
plt.figure()
colors=["lightblue" if (x<max(Catratings["Rating"])) else "blue" for x in Catratings["Rating"]]
plt.bar(height=Catratings["Rating"],x=Catratings.index,color=colors)
plt.xticks(rotation='vertical')
plt.title("Top 10 Categories by Ratings")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.figure()
df['Type'].fillna(str(df['Type'].mode().values[0]), inplace = True) 
df.Type.isnull().sum()

plt.figure()
df.Type.value_counts().plot.pie(y = df.Type, figsize = (8, 10), autopct = '%1.1f%%', title = 'Ratio of Free and Paid apps in the market',label = '');

#Content Rating
plt.figure()
print(df['Content Rating'].unique())
df['Content Rating'].value_counts().plot(kind = 'bar', title = 'Apps by Content Rating', color = my_colors, figsize = (8,4)) 
plt.xlabel('Category')
plt.ylabel('Number of installs')
plt.yscale('log');

df.isnull().sum()
df.fillna(0)

#Heatmap
plt.figure(figsize=(12, 8))
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

df.isnull().sum()
df.dropna(inplace = True)
df.info()
df.drop(labels = ['Current Ver','Android Ver','App'], axis = 1, inplace = True)
df.head()

category_list = df['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)
df.head()

#Label Encoding
le = preprocessing.LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])

le = preprocessing.LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])

df['Price'] = df['Price'].apply(lambda x : x.strip('$'))

df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))

df['Type'] = pd.get_dummies(df['Type'])

df["Size"] = [str(round(float(i.replace("k", ""))/1024, 3)) if "k" in i else i for i in df.Size]

df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
df[df['Size'] == 'Varies with device'] = 0
df['Size'] = df['Size'].astype(float)

df["Size"]

df['new'] = pd.to_datetime(df['Last Updated'])
df['lastupdate'] = (df['new'] -  df['new'].max()).dt.days

#Modelling
x = df.drop(labels=["Rating","Category", "Last Updated", "new"], axis = 1)
y = df['Rating']

#Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

#Apply Algorithms

'''KNN Regression'''
print()
print('------KNN Regression------')
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=50)
knn.fit(x_train, y_train)
accuracy = knn.score(x_test,y_test)
print('KNN Accuracy: ' + str(np.round(accuracy*100, 2)) + '%')
knn_pred = knn.predict(x_test)
mae = mean_absolute_error(y_test, knn_pred)
print('KNN MAE:', mae)
rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
print('KNN RMSE:', rmse)
R2_score = r2_score(y_test, knn_pred)
print('KNN R2_Score:', R2_score)
n_neighbors = np.arange(1, 50, 1)
scores = []
for n in n_neighbors:
    knn.set_params(n_neighbors=n)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neighbors, scores)
plt.show()

# inp=np.array([70769,35,5000000,1,0,1,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# ,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-15
# ]).reshape(1, -1)

# predicted_data = rf_pred.predict(inp)



'''Random Forest Regression'''
print()
print('------Random Forest Regression------')
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test,y_test)
print('RF Accuracy: ' + str(np.round(accuracy*100, 2)) + '%')
rf_pred = regressor.predict(x_test)
mae = mean_absolute_error(y_test, rf_pred)
print('RF MAE:', mae)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print('RF RMSE:', rmse)
R2_score = r2_score(y_test, rf_pred)
print('RF R2_Score:', R2_score)
estimators = np.arange(10, 150, 10)
scores = []
for n in estimators:
    regressor.set_params(n_estimators=n)
    regressor.fit(x_train, y_train)
    scores.append(regressor.score(x_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()
print("max accuracy is: ", max(scores))
print("the number of estimators required to achieve this result: ", estimators[scores.index(max(scores))])


#======================= PREDICTION ==================================


for i in range(0,10):
    if rf_pred[i]> 0 and rf_pred[i]<1:
        # print("============================")
        print()
        print([i],' Rating 0  ')
        print()
        print("============================")
    elif rf_pred[i]> 1 and rf_pred[i]<2:
        # print("============================")
        print()
        print([i],'Rating 1 ')
        print()
        print("============================")
    elif rf_pred[i]> 2 and rf_pred[i]<3:
        # print("============================")
        print()
        print([i],'Rating 2 ')
        print()
        print("============================")
    elif rf_pred[i]> 3 and rf_pred[i]<4:
        # print("============================")
        print()
        print([i],'Rating 3 ')
        print()
        print("============================")
    elif rf_pred[i]> 4 and rf_pred[i]<5:
        # print("============================")
        print()
        print([i],'Rating 4')
        print()
        print("============================")
    else:
        print()
        print([i],'Rating 5')
        print()
        print("============================")






