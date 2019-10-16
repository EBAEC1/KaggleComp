import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from math import sqrt
import pickle


# Description:
# Makes all calls to create linear regression model based on training dataset
# Linear regression model is then dumped to file via pickle
# Predicitons are made by running the test_income_prediction.py file
# Some features are log scaled before entering model
# Some basic outlier dectection is performed on the labels, setting them median values

def main():
    income_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', index_col = False)
    # income_data['Income in EUR'] = abs(income_data['Income in EUR'])

    features = [ 'Profession', 'University Degree', 'Country']
    X = income_data[features]
    X = clean_data(X)


    # Basic extreme outlier cleaning
    income_data.loc[income_data['Income in EUR'] > 340000, 'Income in EUR'] = income_data['Income in EUR'].median()
    income_data.loc[income_data['Income in EUR'] < 5000, 'Income in EUR'] = income_data['Income in EUR'].median()

    y = income_data['Income in EUR']

    X = transform_profession(X)
    X = transform_university_degree(X)
    X = transform_country(X)
    # X = transform_gender(X)
    # X = transform_hair_color(X)
    # X = scale(X,'Age')
    # X = scale(X,'Body Height [cm]')
    # X = scale(X,'Size of City')


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    linear_regression_testing(X_train, y_train, X_test, y_test)



# Description: From basic analysis of data, this function "cleans" the data by
# filling in missing values and ensuring all the columns have values of the same type
# Some of the countries are mapped to others to match the number of countries in the test dataset

# parameters: DataFrame object to be cleaned
# Returns: DataFrame object with updated columns and cell values


def clean_data(income_data):
    # income_data['Year of Record'] = income_data["Year of Record"].fillna('2000')
    income_data['Country'] = income_data['Country'].replace('Liberia', 'Finland')
    income_data['Country'] = income_data['Country'].replace('Congo', 'Finland')
    income_data['Country'] = income_data['Country'].replace('Libya', 'Finland')
    income_data['Country'] = income_data['Country'].replace('Togo', 'Finland')                            # Replacing NaN values in Year of Record with 2000
    # income_data['Gender'] = income_data["Gender"].fillna('male')                                            # Replacing NaN with male
    # income_data['Gender'] = income_data["Gender"].replace('0', 'other')                                    # Acknowledging '0' as a distinct category
    income_data['University Degree'] = income_data["University Degree"].fillna('No')                        # Replacing NaN with No
    income_data['University Degree'] = income_data["University Degree"].replace('0', 'No')                  # Replacing 0 with No
    # income_data['Hair Color'] = income_data["Hair Color"].fillna('Black')
    # income_data['Hair Color'] = income_data["Hair Color"].replace('0', 'Unknown')                         # Possible Problem, p robably to much black hair in the dataset
    # income_data['Age'] = income_data['Age'].fillna(income_data["Age"].mean())                               # Filling NaN
    income_data['Profession'] = income_data['Profession'].fillna('Unemployed')
    return income_data


# Description:
# used to select features to be included in DataFrame
# parameters: dataframe and labels
# Returns: Nothing

def feature_selection(X,y):
    #apply SelectKBest class to extract top 10 best features
    # y = y.astype('int')
    bestfeatures = SelectKBest(score_func=chi2, k=all)
    fit = bestfeatures.fit(X['Age'],y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features


# Description:
# convert categorical Gender values a numeric value
# Uses get_dummies

# parameter: dataframe containing 'Gender'
# Returns: dataframe object with encoded 'Gender' values

def transform_gender(income_data):
    dummies = pd.get_dummies(income_data.Gender)
    merged = pd.concat([income_data,dummies], axis = 'columns')
    final = merged.drop(['Gender','unknown'],axis = 'columns')
    return final

# Description:
# Maps categorical university data to numeric data
# A custom map is created to reflect the ordinal nature of the categorical data
# i.e. more value is given to PhD vs no degree

# parameters: dataframe object containing 'University Degree' data
# Returns: dataframe object with encoded 'University Degree' feature

def transform_university_degree(income_data):
    degree_mapping = {'No':0 , 'Bachelor':1, 'Master':2, 'PhD':3}
    income_data['University Degree'] = income_data['University Degree'].map(degree_mapping)

    return income_data

# Description:
# convert categorical Country values a numeric value
# Uses get_dummies

# parameter: dataframe containing 'Country'
# Returns: dataframe object with encoded 'Country' values


def transform_country(income_data):
    dummies = pd.get_dummies(income_data.Country)
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Country", "Denmark"], axis = 'columns')
    return final

# Description:
# Convert categorical profession data to numerical
# Uses label encoding here as one hot encoding created too many columns
# Tried some dimnesionality reduction to place similiar professions to buckets, but didnt come to be

# parameters: Dataframe containing 'Profession' data
# Returns: Dataframe with updated "Profession" data suitable for linear regression

def transform_profession(income_data):
    le = LabelEncoder()
    income_data['Profession'] = le.fit_transform(income_data['Profession'])
    return income_data

# Description:
# convert categorical Hair color values a numeric value
# Uses get_dummies

# parameter: dataframe containing 'Hair Color'
# Returns: dataframe object with encoded 'Hair Color' values

def transform_hair_color(income_data):
    dummies = pd.get_dummies(income_data["Hair Color"])
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Hair Color", "Black"], axis = 'columns')
    return final

# Description:
# Builds linear regression module based on test and traning data
# Model is trained and evaluated on different data in an effort to stem overfitting
# prints out different metrics: RMSE score, Y-intercept
# Saves model to external file via Pickle which is then loaded in test_income_prediction.py file

# parameters: feature training and test data; label training and test data

def linear_regression_testing(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    print('RMSE ERROR')
    print(sqrt(mean_squared_error(y_test,y_pred)))
    print(model.score(X_test, y_test)*100)
    print(model.intercept_)

    with open('model_pickleV1','wb') as f:
        pickle.dump(model,f)



# Description :
# Used to visul data to manually select best features to use
#  parameters: dataset and label

def lineplot(X,y):
    _, ax = plt.subplots()
    ax.plot(X, y, lw = 2, color = '#539caf', alpha = 1)


# Description:
# transforms a range of values by performing a log transformation
# designed to be utilised for different features

def scale(X, parameter):
    # X[parameter] = StandardScaler().fit_transform(X[parameter].values.reshape(111993,1))
    X[parameter] = np.log(X[parameter])
    return X


if __name__ == "__main__":
    main()







# income_data = income_data.drop(columns=['Instance'])
# income_data = income_data.dropna()                                                                        # Test with dropping missing values: 111993 -> 90400 entries


# print(income_data.isnull().any())
# print(income_data.describe())
# Econding values

#Preprocessing: See imputer manual
#Imputer.fit(X) can specifiy specific column  with [:,:]
#From excel, only categorical data needs to be sorted, no missing numerical data
#Columns to be Encoded: Year of Record[1], Gender[2], Country[4], Profession[6], University Degree[7], Hair Colour[9],
