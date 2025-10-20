import pandas as pd
import sys
from sklearn.model_selection import train_test_split                 # Imports functionality from Scikit Learning
from sklearn.neighbors import KNeighborsClassifier                   # Imports kNN Classifier Implementation from Scikit Learning
from sklearn.neighbors import KNeighborsRegressor                    # Imports kNN Regressor Implementation from Scikit Learning
from sklearn.compose import ColumnTransformer                        # Imports ColumnTransformer
from sklearn.preprocessing import OneHotEncoder                      # Imports OneHotEncoder
from sklearn.preprocessing import MinMaxScaler                       # Imports the MinMaxScaler for Normalization
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Imports Evaluation Metrics from Scikit Learning


# ---------------------------------------------------------------------------------------------

# Display menu
print("**********************************************************")
print("********************* Project Robert *********************")
print("**********************************************************")
print("********************* Data Analysis **********************")
print("*                                                        *")
print("* Please choose one of the following options:            *")
print("*                                                        *")
print("*(1) Load a data set.                                    *")
print("*(2) Train the data set.                                 *")
print("*(3) Evaluation of the data set.                         *")
print("*(4) Simulation of the data set.                         *")
print("**********************************************************")

# Variables set up
userChoice = 0
menuEdit = 1
successLoad = 0  
attemps = 3                                     # Attemps until the code asks if you want to exit
userattempt = 1                                 # Attemp set to check
preprocessor = 0
knn = KNeighborsClassifier(n_neighbors=1)       # Creates a Knn Classifier Object with k=1
knnr = KNeighborsRegressor(n_neighbors=1)       # Creates a Knn Regressor Object with k=1
predictions = 0
features = 0
classes = 0

# Function set up
def loadData():
    global userattempt
    while userattempt == 1:
        try:
            url = input("Insert the name for the dataset: ")
            global successLoad # calling the global variable 
            successLoad = 1    # telling system the load was successfull
            df = pd.read_csv(url)
            print("The dataset was loaded sucessfully.\n")
            print("Overview of some important data information and statistics")
            print("**********************************************************")
            print(df.head(10))
            print("**********************************************************")
            print(df.info())
            print("**********************************************************")
            print(df.describe())
            print("**********************************************************")
            print(df.shape)
            print("**********************************************************")

            # Creates a dataframe using the drop method, which has two parameters:
            # The first parameter tells which labels to remove (Columns Name) or 
            # The second parameter tells whether to remove a row index or a column name. 
            # axis=1 means we want to remove a column.
            global features # calling global variable
            features = df.drop("class",axis=1)

            # Creates a dataframe from just one column:
            global classes # calling global variable
            classes = df["class"]
    
            break

        except: 
            error()
            menu()


def error():
    print ("\nERROR - wrong input!")


def menu():
    global userattempt, menuEdit
    while userattempt == 1:
        try:
            menuEdit = int(input("\n(1) continue or (2) cancel "))
            if 1<= menuEdit <= 2: # Approval of integer (1 or 2)
                break   
            else:
                raise        
        
        except: 
            error()
    
    if menuEdit == 2: # Approval of integer 2
        end()
        pass
        

def end():
    print ("\nThanks for your time and have a great day!\n")
    sys.exit()


def defNumCat():
    categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"] # Define the columns with categorical features
    
    # Updates the Preprocessor to consider the categorical data columns
    global preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

# While loop
while menuEdit == 1:
    while userattempt == 1:
        try:
            userChoice = int(input("\nPlease type in the number of the choosen operation: "))
            if 1<= userChoice <=4:
                break       
            else:
                raise
        except:
             error()
        

    match userChoice:
        case 1:
            # ---------------------------------------------------------------------------------------------
            # 1. LOADING
            loadData()
            menu()


        case 2:
            # ---------------------------------------------------------------------------------------------
            # 2. TRAINING 
            if successLoad == 0:
                print ("There is no data set loaded yet! Please follow further instructions.")                
                loadData()

            if successLoad == 1:
                print("**********************************************************")
                print("******************* Training Algorithms ******************")
                print("*                                                        *")
                print("* Please choose one of the following options:            *")
                print("*                                                        *")
                print("*(1) Split the data into train sets.                     *")
                print("*(2) Split the data into stratified train sets.          *")
                print("**********************************************************")


                for i in range(attemps): # Able to run into an error x attemps until you have the possibility to exit this action
                    try:
                        userChoice = int(input("\nPlease type in the number of the choosen algorithm: "))
                        match userChoice:
                            case 1:
                                # Split the data into train/test sets
                                features_train, features_test, classes_train, classes_test = train_test_split(
                                features, classes, test_size=0.2, random_state=10
                                )

                                defNumCat()
                        
                                # Apply preprocessing to the training set:
                                preprocessed_features_train = preprocessor.fit_transform(features_train)

                                # Apply preprocessing to the test set:
                                preprocessed_features_test = preprocessor.transform(features_test)

                                # Trains this Knn Classifier with the training set obtained previously:
                                knn.fit(preprocessed_features_train, classes_train)

                                predictions = knn.predict(preprocessed_features_test)

                                print ("You have trained this set of data!")
                                break

                            case 2:
                                # Split the data into STRATIFIED train/test sets:
                                strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
                                features, classes, test_size=0.4, random_state=10, stratify=classes
                                )

                                defNumCat()
                        
                                # Trains this Knn Classifier with the training set obtained previously:
                                knn.fit(strat_feat_train, strat_classes_train)

                                predictions = knn.predict(strat_feat_test)
                            
                                print ("You have trained this stratified set of data!")  
                                break

                            case _: 
                                raise      
    
                    except:
                        error()
                
            menu()

        case 3:
            # ---------------------------------------------------------------------------------------------
            # 3. EVALUATION
            pass


        case 4:
            # ---------------------------------------------------------------------------------------------
            # 4. SIMULATION
            pass

        case _:
            error()
            menu()




# ---------------------------------------------------------------------------------------------
# Code Testing
# print (train_test_split)
