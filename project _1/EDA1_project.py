import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import os

print("understanding the dataset")


script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, 'datset.csv')
if not os.path.exists(file_name):
    print(f"error:{file_name} is not found:")
    exit()


df=pd.read_csv(file_name)
print("succesfully loaded")
print(f"Shape pf the dataset:Rows{ df.shape[0]},columns:{df.shape[1]}")


print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())
print(df.info())

median_age=df['Age'].median()
df['Age']=df['Age'].fillna(median_age)
print(median_age)

median_Visits_Per_Month=df["Visits_Per_Month"].median()
df["Visits_Per_Month"]=df["Visits_Per_Month"].fillna(median_Visits_Per_Month)
print(median_Visits_Per_Month)

mean_age=df['Age'].mean()
df['Age']=df['Age'].fillna(mean_age)
print(mean_age)

mean_Visits_Per_Month=df["Visits_Per_Month"].mean()
df["Visits_Per_Month"]=df["Visits_Per_Month"].fillna(mean_Visits_Per_Month)
print(mean_Visits_Per_Month)

plt.figure(figsize=(12,8))
df["Visits_Per_Month"].hist(bins=10,color="purple",edgecolor="black")
plt.title("distribution of visit per month")
plt.xlabel("visit per month")
plt.ylabel("Number of Customers")
plt.show()

correlation=df.corr(numeric_only=True)
print(correlation)

print("Plotting Coorelation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".5f")
plt.title("coorelation Heatmap")
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen')
plt.title("Boxplot of customer age")
plt.xlabel('Age')
plt.show()

print("find the outliers in age")
outliers=df[df["Age"]>100]
print("find the outliers(s):")
print(outliers)






