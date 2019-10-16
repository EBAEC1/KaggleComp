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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score as acc
from math import sqrt
import pickle


def main():
    income_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', index_col = False)
    # income_data['Income in EUR'] = abs(income_data['Income in EUR'])

    # features = [ 'Country', 'Age', 'Year of Record', 'Profession',
    #    'Size of City',  'University Degree', 'Wears Glasses',
    #     'Body Height [cm]']

    # profession_list = {'manager','developer','engineer','captain','scientist','officer','care','curer','ambassador','reviewer','filler','trainer','patrol','taker','dental','loaders','official','pathologist','erector',
    #                     'consultant','assistant','technician','scientist','liaison','supervisor','specialist','clerk','advisor','inspector', 'roof','resources','vendor','evaluator','tender','relations','reviewer',
    #                     'accountant','counsel','lead','nurse','agent','intern','planner','associate','trainee','coordinator','teacher','guard','investigator', 'assembler','conductor','server','microbiologist',
    #                     'painter','administrator','physician','mechanic','researcher','cleaner','architect','attendant','electrician','auditor','representative', 'cashier','broker','commissioner','maintainer',
    #                     'designer','tax','worker','doctor','contractor','examiner','guide','buyer','dealer','expert','facilitator','therapist','repairer','aide','attorney','marine','groundskeeper','policeman',
    #                     'messenger','photographer','installer','labourer','driver','fellow','collector','staff','controller','operator','analyst','finisher','postal','reader','promoter','animal','laborer',
    #                     'art','owner','management','health','writer','mason','searcher','instructor','caller','betting','grinder','cook','servicer','head','superintendent','publisher',
    #                     'poster','chemist','building','coach','adjustor','support','pruner','handler','fitter','handler','diver','seller','pair','crew','binder','bar','commander','minister','warden','processor','advocate',
    #                     'member','machinist','processor','review','adjuster','estimator','hygienist','detective','salesperson','builder','service','fighter'}
    #
    # hi_list = {'manager','developer','consultant','physician','Anaesthetist','doctor','veterinarian','programmer','accountant','investor','attorney','c-suite','psychologist','judge','prosecutor','mathematician','dentist','statistican,''Anaesthetist','economist','Actor'}
    #
    # mid_list = {'engineer','captain','scientist','microbiologist','specialist','Biologist','officer','technician','lead','therapist','fellow','architect','counsel','expert','analyst','associate','optician','Audiologist','resources','animal','contractor',
    #             'paralegal','sergeant','superintendent','producer','podiatrist','optometrist','salesperson','jeweler', 'hygienist','estimator','magistrate','statistican','geologist','guard','reporter','processor','commissioner','dietitian','Botanist','Astrologer','translator','Anthropologist','musician',
    #             'advisor','supervisor','commander','expert','physicist','pharmacist','chemist','entertainer','cosmetologist','tax','Astronomer','nutritionist','detective','legislator','inspector','liason','owner','support','art','management','health','writer','logistician','photogrammetrist','musician'
    #             'jeweller','assistant','editor','ombudsperson','publisher','statistician','dental','broker','Actress','pathologist','postmaster','sociologist','warden','interpreter','minister','Botanist','Bodyshop','Baker','adjuster','adjustor','Archaeologist','Arbitrator','Arborist','author','actuary','criminalist','Acupuncturist'}
    #
    # mid_lo_list = {'administrator','mechanic','researcher','performer','attendant','librarian','electrician','auditor','representative','Beautician','breeder','liaison','engraver','carpenter','caoch','trapper','Arborist','libraian','member','forester','hunter','libraian','evaluator','official','tender','mason','caster','Bookmaker','reviewer', 'distributor','Book-Keeper'
    #                 'designer','worker','examiner','guide','buyer','dealer','facilitator','repairer','aide','crew','typist','etchers','teacher','agent','interviewer','head','fighter','marine','assessor','rigger','timekeeper','relations','seller','custodian','poster','dispatcher','hairstylist','presser','machinist','tailor','host',
    #                 'messenger','photographer','installer','labourer','driver','welder','collector','staff','controller','operator','finisher','telemarketer','trainer','patrol','roof','conductor','policeman','announcer','bookkeeper','chef','Auctioneer','jailer','distributor','courier','Airman','Balloonist'
    #                 'plasterer','paramedic','reader','service','investigator','planner', 'pair', 'nurse','appraiser','rancher','sailor','plumber','promoter','measurer','singer','Balloonist','boilermaker','dressmaker','choreographer','translator','searcher','instructor','umpire','farmer','surveyor','ambassador','trainee',}
    #
    # lo_list = {'painter','cleaner','bar','attendant','coordinator','drafter','groundskeeper','budget','millwright','patternmaker','sewer','review','roustabout','clerk','curer','builder','diver','hostler','maid','packer','unemployed','intern','greeter','paperhanger','deckhand','grader','usher','yardmaster','sorter','gardener','packager','taker','vendor','cashier','concierge','cook',
    #                 'pipelayer','Bricklayer','building','molder','receptionist','advocate','server','erector','plasterer','bellhop','helper',
    #                 'designer','worker','examiner','guide','buyer','facilitator','repairer','aide','hairdresser','caller','betting','grinder','servicer','filler','reviewer','assembler','loaders','laborer','handler','Blacksmith','cabinetmaker','curator','binder','pruner','clergy','fisher','bailiff','builing','Book-Keeper','upholsterer','glazier',
    #                 'messenger','photographer','installer','chauffeur','labourer','demonstrator','driver','collector','staff','controller','finisher','janitor','sampler','maintainer','dancer','waiter','taper','Brewer','waitress','teller','Barber','fitter','Armourer','Occupations','Archivist', 'locksmith', 'helper','weigher','coach','Almoner','care',
    #                 'dishwasher','butcher','tender','postal'}
    #
    # income_data['Profession'] = income_data['Profession'].fillna('unemployed')
    #
    # profession_mask = income_data['Profession']
    # income_data = income_data.drop(columns=["Profession"])
    #
    #
    # profession_mask = set_mask(profession_mask, profession_list)
    # profession_mask = manual_set(profession_mask)
    # profession_mask = hi_set(profession_mask, hi_list)
    # profession_mask = mid_set(profession_mask, mid_list)
    # profession_mask = mid_lo_set(profession_mask, mid_lo_list)
    # profession_mask = lo_set(profession_mask, lo_list)
    #
    # profs = profession_mask.unique()
    # # pd.DataFrame(profs, columns=['Profession']).to_csv('Profs.csv')
    # # print(profs)
    #
    # income_data['Profession'] = profession_mask
    # print(income_data['Profession'])
    #
    # unique = income_data['Profession'].unique()
    #
    # prediction = pd.DataFrame(unique, columns=['Profession']).to_csv('Profession.csv')


    features = [ 'Profession', 'University Degree', 'Country']
    # #
    X = income_data[features]
    # #
    X = clean_data(X)
    # #
    income_data.loc[income_data['Income in EUR'] > 340000, 'Income in EUR'] = income_data['Income in EUR'].median()
    income_data.loc[income_data['Income in EUR'] < 5000, 'Income in EUR'] = income_data['Income in EUR'].median()
    #
    y = income_data['Income in EUR']
    # # print(income_data['Income in EUR'].describe())
    # # lineplot(income_data['University Degree'], y)
    #
    X = transform_profession(X)
    X = transform_university_degree(X)
    X = transform_country(X)
    # X = scale(X,'Age')
    # X = scale(X,'Body Height [cm]')
    # # X = scale(X,'Size of City')
    #
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    random_Forest(X_train, y_train, X_test, y_test)


# Year of record,  Age, Country, Size of city, Profession, desgree, wears glasses, hair colour,

def clean_data(income_data):
    # income_data['Year of Record'] = income_data["Year of Record"].fillna('2000')
    # income_data['Country'] = income_data['Country'].replace('Liberia', 'Finland')
    # income_data['Country'] = income_data['Country'].replace('Congo', 'Finland')
    # income_data['Country'] = income_data['Country'].replace('Libya', 'Finland')
    # income_data['Country'] = income_data['Country'].replace('Togo', 'Finland')                            # Replacing NaN values in Year of Record with 2000
    # income_data['Country'] = income_data['Country'].replace('Czechia', 'Finland')                            # Replacing NaN values in Year of Record with 2000
    # income_data['Country'] = income_data['Country'].replace('Gabon', 'Finland')                            # Replacing NaN values in Year of Record with 2000

    # # income_data['Gender'] = income_data["Gender"].fillna('male')                                            # Replacing NaN with male
    # # income_data['Gender'] = income_data["Gender"].replace('0', 'other')                                    # Acknowledging '0' as a distinct category
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
    profession_mask.loc[profession_mask.str.contains('programmer', case=False)] = 'developer'
    profession_mask.loc[profession_mask.str.contains('tech', case=False)] = 'technician'
    profession_mask.loc[profession_mask.str.contains('investment', case=False)] = 'investor'
    profession_mask.loc[profession_mask.str.contains('director', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('chief', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('executive', case=False)] = 'c-suite'
    profession_mask.loc[profession_mask.str.contains('data', case=False)] = 'developer'

    return profession_mask

def hi_set(profession_mask, job_list):
    for job in job_list:
        profession_mask.loc[profession_mask.str.contains(job, case=False)] = 'high'

    return profession_mask


def mid_set(profession_mask, job_list):
    for job in job_list:
        profession_mask.loc[profession_mask.str.contains(job, case=False)] = 'mid'

    return profession_mask

def mid_lo_set(profession_mask, job_list):
    for job in job_list:
        profession_mask.loc[profession_mask.str.contains(job, case=False)] = 'midlo'

    return profession_mask

def lo_set(profession_mask, job_list):
    for job in job_list:
        profession_mask.loc[profession_mask.str.contains(job, case=False)] = 'lo'

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
    # encoder = LabelEncoder()
    # degree = income_data['University Degree']
    # encoded_degrees = encoder.fit_transform(degree['No':0, 'Bachelor':1, 'Master':2, 'PhD':3])
    # print(encoded_degrees)
    # print(encoder.classes_)
    degree_mapping = {'No':0 , 'Bachelor':1, 'Master':2, 'PhD':3}
    income_data['University Degree'] = income_data['University Degree'].map(degree_mapping)

    # dummies = pd.get_dummies(income_data['University Degree'])
    # merged = pd.concat([income_data, dummies], axis = 'columns')
    # final = merged.drop(["University Degree", "No"], axis = 'columns')
    return income_data

def transform_country(income_data):
    dummies = pd.get_dummies(income_data.Country)
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Country", "Denmark"], axis = 'columns')
    return final

def transform_profession(income_data):
    # dummies = pd.get_dummies(income_data.Profession)
    # merged = pd.concat([income_data, dummies], axis = 'columns')
    # final = merged.drop(["Profession", "tour guide"], axis = 'columns')
    le = LabelEncoder()
    income_data['Profession'] = le.fit_transform(income_data['Profession'])
    print(income_data['Profession'])

    return income_data

def transform_hair_color(income_data):
    dummies = pd.get_dummies(income_data["Hair Color"])
    merged = pd.concat([income_data, dummies], axis = 'columns')
    final = merged.drop(["Hair Color", "Black"], axis = 'columns')
    return final

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

def linear_regression(X,y):
    model = linear_regression()
    model.fit(X,y)
    y_pred = model.predict(X)
    print(pd.DataFrame({'Actual': y, 'Prediction': y_pred}))

def lineplot(X,y):
    _, ax = plt.subplots()
    ax.plot(X, y, lw = 2, color = '#539caf', alpha = 1)

def scale(X, parameter):
    # X[parameter] = StandardScaler().fit_transform(X[parameter].values.reshape(111993,1))
    X[parameter] = np.log(X[parameter])
    return X

def random_Forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators = 16)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    print('RMSE ERROR')
    print(sqrt(mean_squared_error(y_test,y_pred)))
    print(model.score(X_test, y_test)*100)

    with open('ranndomForest_pickleV1','wb') as f:
        pickle.dump(model,f)


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
