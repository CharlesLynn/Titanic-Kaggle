import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_genderclassmodel = pd.read_csv('data/genderclassmodel.csv')
#df_gendermodel = pd.read_csv('data/gendermodel.csv')

def print_nulls(df):
	#EDA for Null Values
	print "Number of {}: %{}".format(len(df))
	for feature in df.columns:	
		print "{} Null: %{}".format(sum(df['feature'].isnull())/float(len(df)))
	

# Number of Passengers: 891
# Pclass Null: 0
# Sex Null: 0
# Age Null: 177
# SibSp Null: 0
# Ticket Null: 0
# Fare Null: 0
# Cabin Null: 687
# Embarked Null: 2


def process_df(df):
	
	#Drop Features
	df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

	#Fill Nulls
	df.loc[(df.Age.isnull()), 'Age'] = df['Age'].dropna().median()
	
	# All the missing Fares -> assume median of their respective class
	if len(df.Fare[ df.Fare.isnull() ]) > 0:
	    median_fare = np.zeros(3)
	    for f in range(0,3):                                              # loop 0 to 2
	        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
	    for f in range(0,3):                                              # loop 0 to 2
	        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

	median_age = df['Age'].dropna().median()
	if len(df.Age[ df.Age.isnull() ]) > 0:
	    df.loc[ (df.Age.isnull()), 'Age'] = median_age



	# Cabin to Letter and Number ex. A203 ->  A, 203
	df['Cabin_Letter'] = df.Cabin[df['Cabin'].notnull()].apply(lambda x: x[0])
	#df['Cabin_Number'] = df.Cabin[df['Cabin'].notnull()].apply(lambda x: x[1:].split(' ')[0])
	df.drop('Cabin', axis=1, inplace=True)
	df['Sex'] = df['Sex'].map({'female' : 0, 'male' : 1})

	#Dummy and drop old column
	df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1).drop('Embarked', axis=1)
	df = pd.concat([df, pd.get_dummies(df['Pclass'])], axis=1).drop('Pclass', axis=1)
	df = pd.concat([df, pd.get_dummies(df['Cabin_Letter'])], axis=1).drop('Cabin_Letter', axis=1)


	return df


df_train = process_df(df_train)

X_train = df_train.drop(['Survived'], axis=1)
y_train = df_train['Survived']
X_test = process_df(df_test)
y_test = df_genderclassmodel['Survived']

X_train.drop(X_train.columns - X_test.columns, axis=1, inplace=True)
print len(X_train.columns), len(X_test.columns)

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




# log_model = LogisticRegression()
# log_model.fit(X_train, y_train)
# print log_model.score(X_test, y_test)

# forest_model = RandomForestClassifier(n_estimators=20, oob_score=False)
# forest_model.fit(X_train, y_train)
# print forest_model.score(X_test, y_test)

