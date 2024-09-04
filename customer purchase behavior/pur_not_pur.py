import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import joblib

data = pd.read_csv("customer_purchase_data.csv")
data.sort_values(by="Age", inplace=True)
#print(data.head())

#plt.scatter(data["Age"],data["PurchaseStatus"], color="red")
#plt.show()

print(data.isnull().sum())
print(data.duplicated().sum())
data = data.drop_duplicates()
print(data.duplicated().sum())

print(data.head(5))
print(data.shape)
print(data.info())

features = data.drop("PurchaseStatus", axis=1)
print(features)
target = data["PurchaseStatus"]
#print(target)

mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)
print(nfeatures)
joblib.dump(mms, 'scaler.pkl')

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

model = GaussianNB()
model.fit(x_train, y_train)

print(x_test)
print(y_test)
print(model.predict(x_test))

cm = confusion_matrix(y_test, model.predict(x_test))
print(cm)

cr = classification_report(y_test, model.predict(x_test))
print(cr)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Using Seaborn Heatmap')
plt.show()

print(data.head(2))
#data1 = [[40,1,66120.26794,8,0,30.57,0,5]]
#data2 = [[52,0,40534.12697,1,1,54.29,0,2]]
#data3 = [[20,1,23579.77358,4,2,38.24,0,5]]
#data4 = [[27,1,127821.3064,11,2,31.63,1,0]]
#data5 = [[40,1,57363.24754,7,4,12.21,0,0]]
#data6 = [[55,0,81936.80822,6,1,56.94,1,4]]
#data7 = [[60,1,122703.9567,20,4,36.57,0,4]]
#data8 = [[50,0,4417925,13,0,25.35,1,4]]
age = int(input("enter age: "))
sex = int(input("0 for male 1 for female: "))
ai = int(input("enter income: "))
purchases = int(input("enter purchases: "))
pc = int(input("0 electronics 1 clothing 2 home goods 3 beauty 4 sports: "))
tsow = float(input("enter time spent on website: "))
lpm = int(input("loylaty promgram number 0 for no 1 for yes: "))
da = int(input("enter discount 0 to 5: "))
d = [[age,sex,ai,purchases,pc,tsow,lpm,da]]
nd = mms.transform(d)
ans = model.predict(nd)
print(ans)
ans = ans[0]
if ans == 0:
	print("no purchase")
else:
	print("purchase")
'''
from pickle import *
f = open("cpb1.pkl","wb")
dump(model, f)
f.close()
'''



