import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split                       # Imports functionality from Scikit Learning
from sklearn.tree import DecisionTreeClassifier                            # Imports Decision Tree Classifier from Scikit Learning
from sklearn.neighbors import KNeighborsClassifier                         # Imports kNN Classifier Implementation from Scikit Learning
from sklearn.compose import ColumnTransformer                              # Imports ColumnTransformer
from sklearn.preprocessing import OneHotEncoder                            # Imports OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score  # Imports Evaluation Metrics from Scikit Learning
from sklearn.metrics import mean_absolute_error, mean_squared_error        # Imports Evaluation Metrics from Scikit Learning
from sklearn.tree import DecisionTreeClassifier


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
dt = DecisionTreeClassifier(random_state=77)    # Creates a Decision tree classifier object
predictions = 0
features = 0
classes = 0
modelTrained = 0
filename = ""

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
                print("*(1) Split the data and train with the lazy learner - KNN*")
                print("*(2) Split the data and train with Decision Tress        *")
                print("**********************************************************")

                # -----------------------------------------------------------------------------------------
                # Split the data into STRATIFIED train/test sets:
                strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
                features, classes, test_size=0.2, random_state=10, stratify=classes
                )

                # Define the columns with categorical features
                categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"] 
                
                # Updates the Preprocessor to consider the categorical data columns
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(), categorical_features)
                    ]
                )

                # Apply preprocessing to the training set:
                preprocessed_strat_feat_train = preprocessor.fit_transform(strat_feat_train)

                # Apply preprocessing to the test set:
                preprocessed_strat_feat_test = preprocessor.transform(strat_feat_test)

                # -----------------------------------------------------------------------------------------


                for i in range(attemps): # Able to run into an error x attemps until you have the possibility to exit this action
                    try:
                        userChoice = int(input("\nPlease type in the number of the choosen algorithm: "))
                        match userChoice:
                            case 1:
                                # Trains this Knn Classifier with the training set obtained previously:
                                dt.fit(preprocessed_strat_feat_train, strat_classes_train)

                                predictions = knn.predict(preprocessed_strat_feat_test)

                                modelTrained = 1

                                print ("You have trained this stratified set of data with a KNN classifier!")
                                break

                            case 2:
                                # Trains this Knn Classifier with the training set obtained previously:
                                knn.fit(preprocessed_strat_feat_train, strat_classes_train)

                                predictions = dt.predict(preprocessed_strat_feat_test)

                                modelTrained = 1

                                print ("You have trained this stratified set of data with an Dessicion Tree classifier!") 
                                break

                            case _: 
                                raise      
    
                    except:
                        error()
                
            menu()

        case 3:
            # ---------------------------------------------------------------------------------------------
            # 3. EVALUATION
            if modelTrained == 1:
                userChoice = int(input("Would you like to load a file for evaluation? (1)Yes (2)No"))
                if userChoice == 1:
                    evalfile = input("Type the name of the file: ")
                    df2 = pd.read_csv(evalfile)
                    print(df2.head(5))
                    
                    
                    #Prints the accuracy:
                    print("Accuracy:", accuracy_score(strat_classes_test, predictions))
                    print("Precision:", precision_score(strat_classes_test, predictions, average='weighted'))
                    print("Recall:", recall_score(strat_classes_test, predictions, average='weighted'))
                
                elif userChoice == 2:
                    #Prints the accuracy:
                    print("Accuracy:", accuracy_score(strat_classes_test, predictions))
                    # print("Precision:", precision_score(strat_classes_test, predictions))
                    # print("Recall:", recall_score(strat_classes_test, predictions))
                # Savings results
                userChoice = int(input("Would you like to save the results? (1)Yes (2)No"))
                if userChoice == 1:
                    filename = input("Please type in the name of the document: ")
                    with open(filename, "w") as f:
                        f.write(f"{filename}\n")
                        f.write(f"Accuracy: {accuracy_score}")
                        f.write(f"Precision: {precision_score}")
                        f.write(f"Recall: {recall_score}")
                        f.write(f"Predictions:\n{predictions}")

                elif userChoice == 2:
                    print ("You have not saved these results!")
                else:
                    error()
            else:
                print("You have not trained any models yet!")
                menu()

        case 4:
            # ---------------------------------------------------------------------------------------------
            # 4. SIMULATION
            print("Please enter the values for an object you would to simulate.")
            
            # maybe using a class here would be kinda usefull???


        case _:
            error()
            menu()




# ---------------------------------------------------------------------------------------------
# Code Testing
# print (train_test_split)
