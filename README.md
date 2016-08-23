#Titanic-Kaggle
Titanic Kaggle dataset Model Comparison. The goal of this challenge is to use the provided dataset to predict weather a passenger survived the sinking of the Titanic. I decided to use this simple dataset to a compare of the performance of various algorithms, including logistic regression.

##Features
- VARIABLE DESCRIPTIONS:
- survival        Survival (0 = No; 1 = Yes)
- pclass          Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- name            Name
- sex             Sex
- age             Age
- sibsp           Number of Siblings/Spouses Aboard
- parch           Number of Parents/Children Aboard
- ticket          Ticket Number
- fare            Passenger Fare
- cabin           Cabin
- embarked        Port of Embarkation
- Features included in model: pclass, sex, age, sibsp, parch, fare, cabin (letter prefix), embarked.

##Algorithms
- LogisticRegression : 93.540670 %
- RandomForest : 80.382775 %
- GradientBoosting : 89.952153 %
- GNB : 67.942584 %
- DecisionTree : 84.928230 %
- AdaBoost : 90.669856 %

- Winner algorithm is LogisticRegression with a 93.540670 % success


##Conclusion
This dataset initially lends itself to a decision a tree type model, because many of the features are categorical and decision trees work appropriately with not-continuous values, while liner models would assume that Port "2" is two times greater than Port "1". However, I wanted to see how a liner model would preform and dummying the categorical features into multiple binary features was the obvious solution. It turned out that LogisticRegression* performed better than any of the decision tree algorithms. It is important to remember that the simplest algorithms sometimes perform the best.  (*Note: These models were not precisely tuned) 