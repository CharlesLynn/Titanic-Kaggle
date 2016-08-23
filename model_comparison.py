import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_genderclassmodel = pd.read_csv('data/genderclassmodel.csv')
#df_gendermodel = pd.read_csv('data/gendermodel.csv')

def print_nulls(df):
	#EDA for Null Values
	print "Number of passengers: {}".format(len(df))
	for feature in df.columns:	
		print "{} Null: {}".format(feature, round(sum(df[feature].isnull()), 2))
	print '\n'

def process_df(df):
	
	#Drop Features
	df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

	#Fill missing Ages with median
	df.loc[(df.Age.isnull()), 'Age'] = df['Age'].dropna().median()
	
	#Fill missing Fares with the median of their respective class
	median_fare = np.zeros(3)
	for f in range(0,3):
		median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
	for f in range(0,3):
		df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

	# Cabin to Letter and Number ex. A203 ->  A
	df['Cabin_Letter'] = df.Cabin[df['Cabin'].notnull()].apply(lambda x: x[0])
	df.drop('Cabin', axis=1, inplace=True)
	
	#Sex to binary values
	df['Sex'] = df['Sex'].map({'female' : 0, 'male' : 1})

	#Dummy and drop old columns
	df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1).drop('Embarked', axis=1)
	df = pd.concat([df, pd.get_dummies(df['Pclass'])], axis=1).drop('Pclass', axis=1)
	df = pd.concat([df, pd.get_dummies(df['Cabin_Letter'])], axis=1).drop('Cabin_Letter', axis=1)


	return df

print_nulls(df_train)
df_train = process_df(df_train)



X_train = df_train.drop(['Survived'], axis=1)
y_train = df_train['Survived']
X_test = process_df(df_test)
y_test = df_genderclassmodel['Survived']

#Drop non-consistent features created in dummying .
X_train.drop(X_train.columns - X_test.columns, axis=1, inplace=True)


#Algorithm comparison
algorithms = {
        "LogisticRegression":  LogisticRegression(),
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
        "RandomForest": ske.RandomForestClassifier(n_estimators=50),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
        "GNB": GaussianNB()
    }


results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))



