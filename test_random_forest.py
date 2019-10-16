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


def main():
    income_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv', index_col = False)
    # income_data['Income in EUR'] = abs(income_data['Income in EUR'])
    # y = income_data['Income in EUR']

    # features = [ 'Age', 'Country', 'Profession',
    #    'Size of City',  'University Degree', 'Wears Glasses',
    #     'Body Height [cm]']
    #

    profession_list = {'manager','developer','programmer','engineer','captain','scientist','officer',
                        'consultant','assistant','technician','scientist','liaison','supervisor','specialist','clerk','advisor','inspector',
                        'accountant','counsel','lead','nurse','agent','intern','planner','associate','trainee','coordinator','teacher','guard','investigator',
                        'painter','administrator','physician','mechanic','researcher','cleaner','architect','attendant','electrician','auditor','representative',
                        'designer','worker','doctor','contractor','examiner','guide','buyer','dealer','expert','facilitator','therapist','repairer','aide','attorney',
                        'messenger','photographer','installer','labourer','driver','fellow','collector','staff','controller','operator','analyst'}

    income_data['Profession'] = income_data['Profession'].fillna('Unemployed')

    profession_mask = income_data['Profession']
    income_data = income_data.drop(columns=["Profession"])


    profession_mask = set_mask(profession_mask, profession_list)
    profession_mask = manual_set(profession_mask)

    income_data['Profession'] = profession_mask




    features = [ 'Profession', 'University Degree', 'Country']
    X = income_data[features]



    X = clean_data(X)

    X = transform_university_degree(X)
    X = transform_country(X)
    X = transform_profession(X)
    # # X = transform_hair_color(X)
    # X = scale(X,'Age')
    # X = scale(X,'Body Height [cm]')
    # X = scale(X,'Size of City')

    with open('model_pickleV1','rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    prediction = pd.DataFrame(predictions, columns=['predicitons']).to_csv('prediction.csv')

# Year of record,  Age, Country, Size of city, Profession, degree, wears glasses, hair colour,

def clean_data(income_data):
    # income_data['Year of Record'] = income_data["Year of Record"].fillna('2000')                            # Replacing NaN values in Year of Record with 2000
    # income_data['Gender'] = income_data["Gender"].fillna('male')                                            # Replacing NaN with male
    # income_data['Gender'] = income_data["Gender"].replace('0', 'other')                                    # Acknowledging '0' as a distinct category
    income_data['University Degree'] = income_data["University Degree"].fillna('No')                        # Replacing NaN with No
    income_data['University Degree'] = income_data["University Degree"].replace('0', 'No')                  # Replacing 0 with No
    # # income_data['Hair Color'] = income_data["Hair Color"].fillna('Black')
    # # income_data['Hair Color'] = income_data["Hair Color"].replace('0', 'Unknown')                                    # Possible Problem, p robably to much black hair in the dataset
    # income_data['Age'] = income_data['Age'].fillna(income_data["Age"].mean())                               # Filling NaN
    income_data['Profession'] = income_data['Profession'].fillna('Unemployed')
    return income_data                             # Filling NaN

def set_mask(profession_mask, job_list):
    for job in job_list:
        profession_mask.loc[profession_mask.str.contains(job, case=False)] = job

    return profession_mask

def manual_set(profession_mask):
    profession_mask.loc[profession_mask.str.contains('surgeon', case=False)] = 'doctor'
    profession_mask.loc[profession_mask.str.contains('programmer', case=False)] = 'programmer'
    profession_mask.loc[profession_mask.str.contains('tech', case=False)] = 'technician'
    profession_mask.loc[profession_mask.str.contains('investment', case=False)] = 'investor'
    profession_mask.loc[profession_mask.str.contains('director', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('chief', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('executive', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('data', case=False)] = 'developer'

    return profession_mask


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


def transform_gender(income_data):
    dummies = pd.get_dummies(income_data.Gender)
    merged = pd.concat([income_data,dummies], axis = 'columns')
    final = merged.drop(['Gender','unknown'],axis = 'columns')
    return final

def transform_university_degree(income_data):
    degree_mapping = {'No':0 , 'Bachelor':1, 'Master':2, 'PhD':3}
    income_data['University Degree'] = income_data['University Degree'].map(degree_mapping)
    return income_data

def transform_country(income_data):
    dummies = pd.get_dummies(income_data.Country)
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Country", "Denmark"], axis = 'columns')
    return final

def transform_profession(income_data):
    le = LabelEncoder()
    income_data['Profession'] = le.fit_transform(income_data['Profession'])
    return income_data

def transform_hair_color(income_data):
    dummies = pd.get_dummies(income_data["Hair Color"])
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Hair Color", "Black"], axis = 'columns')
    return final

def linear_regression_testing(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.intercept_)
    y_pred = model.predict(X_test)
    print(pd.DataFrame({'Actual': y_test, 'Predicited': y_pred}))
    print('RMSE ERROR')
    print(sqrt(mean_squared_error(y_test,y_pred)))

def linear_regression(X,y):
    model = linear_regression()
    model.fit(X,y)
    y_pred = model.predict(X)
    print(pd.DataFrame({'Actual': y, 'Prediction': y_pred}))

def lineplot(X,y):
    _, ax = plt.subplots()
    ax.plot(X, y, lw = 2, color = '#539caf', alpha = 1)

def scale(X, parameter):
    X[parameter] = StandardScaler().fit_transform(X[parameter].values.reshape(73230,1))
    return X


if __name__ == "__main__":
    main()
# income_data = income_data.drop(columns=['Instance'])
# income_data = income_data.dropna()                                                                        # Test with dropping missing values: 111993 -> 90400 entries


# print(X_train)

# print(income_data.isnull().any())
# print(income_data.describe())
# Econding values

#Preprocessing: See imputer manual
#Imputer.fit(X) can specifiy specific column  with [:,:]
#From excel, only categorical data needs to be sorted, no missing numerical data
#Columns to be Encoded: Year of Record[1], Gender[2], Country[4], Profession[6], University Degree[7], Hair Colour[9],
