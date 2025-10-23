import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split                       # Imports functionality from Scikit Learning
from sklearn.tree import DecisionTreeClassifier                            # Imports Decision Tree Classifier from Scikit Learning
from sklearn.neighbors import KNeighborsClassifier                         # Imports kNN Classifier Implementation from Scikit Learning
from sklearn.compose import ColumnTransformer                              # Imports ColumnTransformer
from sklearn.preprocessing import OneHotEncoder                            # Imports OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score  # Imports Evaluation Metrics from Scikit Learning

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
print("*(5) Exit the menu.                                      *")
print("**********************************************************")

# Variables set up
userChoice = 0
menuEdit = 1
successLoad = 0  
attemps = 3                                     # Attemps until the code asks if you want to exit
userattempt = 1                                 # Attemp set to check
testPerc = None
preprocessor = 0
algoChoice = 0
knn = KNeighborsClassifier(n_neighbors=1)       # Creates a Knn Classifier Object with k=1
dt = DecisionTreeClassifier(random_state=77)    # Creates a Decision tree classifier object
predictions = 0
features = 0
classes = 0
modelTrained = 0
filename = ""
objectAdd = 1

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
            menuEdit = int(input("\n(1) continue or (2) exit "))
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
            if 1<= userChoice <=5:
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
                print("*(2) Split the data and train with Decision Trees        *")
                print("**********************************************************")

                # -----------------------------------------------------------------------------------------
                # Ask the user of the test/train percentage set up
                print("To set up the test/train set you have to decide about the mixture, e.g. 20/80 or 50/50")
                
  
                for i in range(attemps):
                    try:
                        testPerc = int(input("Please type in the percentage of the testing data set: "))/100
                        if 0 <= testPerc <= 1:
                            break
                        else:
                            testPerc = None
                            raise
                    except:
                        error()

                if testPerc is None:
                    testPerc = 0.2
                    print("\nYou have typed in the wrong input multiple times.")
                    print(f"We will be setting the value to {testPerc*100}% for you!")

                    

                # Split the data into STRATIFIED train/test sets:
                strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
                features, classes, test_size=testPerc, random_state=10, stratify=classes
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
                                knn.fit(preprocessed_strat_feat_train, strat_classes_train)
                                predictions = knn.predict(preprocessed_strat_feat_test)
                                algoChoice = 1
                                modelTrained = 1

                                print ("You have trained this stratified set of data with a KNN classifier!")
                                break

                            case 2:
                                # Trains this Knn Classifier with the training set obtained previously:
                                dt.fit(preprocessed_strat_feat_train, strat_classes_train)
                                predictions = dt.predict(preprocessed_strat_feat_test)
                                algoChoice = 2
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
                
                print("**********************************************************")
                print("*********************** Evaluation ***********************")
                print("*                                                        *")
                print("* Please choose one of the following options:            *")
                print("*                                                        *")
                print("*(1) Load a file for evaluating the trained data.        *")
                print("*(2) Do not load a file and work with another strategy.  *")
                print("**********************************************************")


                userChoice = int(input("Please type in the number of the choosen evaluation: "))
                if userChoice == 1:
                    evalfile = input("Type the name of the file: ")
                    df2 = pd.read_csv(evalfile)
                    
                    features2 = df2.drop("class",axis=1)
                    classes2 = df2["class"]
                    preprocessed_features2 = preprocessor.transform(features2)
                    if algoChoice == 1:
                        predictions2 = knn.predict(preprocessed_features2)

                    elif algoChoice == 2:
                        predictions2 = dt.predict(preprocessed_features2)
                    
                    else: 
                        error()

                    #Prints the accuracy:
                    print("Accuracy:", accuracy_score(classes2, predictions2))
                    print("Precision:", precision_score(classes2, predictions2, average='weighted'))
                    print("Recall:", recall_score(classes2, predictions2, average='weighted'))
                
                elif userChoice == 2:
                    #Prints the accuracy:
                    print("Accuracy:", accuracy_score(strat_classes_test, predictions))
                    print("Precision:", precision_score(strat_classes_test, predictions, average='weighted'))
                    print("Recall:", recall_score(strat_classes_test, predictions, average='weighted'))
                
                
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
                    print ("You have successfully saved these results!")

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
            if modelTrained == 1:
                print("**********************************************************")
                print("*********************** Simulation ***********************")
                print("*                                                        *")
                print("* Please enter the values for an object you would like   *")
                print("* to simulate.                                           *")
                print("**********************************************************")


                class simu():

                    def __init__(self, buying, maint, doors, persons, lug_boot, safety):
                        self.buying = buying
                        self.maint = maint
                        self.doors = doors
                        self.persons = persons
                        self.lug_boot = lug_boot
                        self.safety = safety
                    
                    def string(self):
                        return f"{self.buying}, {self.maint}, {self.doors}, {self.persons}, {self.lug_boot}, {self.safety}"

                    def printObj(self):
                        return(self.buying, self.maint, self.doors, self.persons, self.lug_boot)

                simuList = []

                while objectAdd == 1:
                    print("Please enter the values for an object you would like to simulate.")
                    while objectAdd == 1:
                        buying = input(f"How is the buying?(vhigh, high, med, low): ")
                        if buying in ["vhigh", "high", "med", "low"]:
                            break
                        else:
                            error()
                    while objectAdd == 1:
                        maint = input(f"How is the maintenance?(vhigh, high, med, low): ")
                        if maint in ["vhigh", "high", "med", "low"]:
                            break
                        else:
                            error()
                    while objectAdd == 1:
                        doors = input(f"How many doors are existing?(2, 3, 4, 5more): ")
                        if doors in ["2", "3", "4", "5more"]:
                            break
                        else:
                            error()
                    while objectAdd == 1:
                        persons = input(f"How many persons fit in?(2, 4, more): ")
                        if persons in ["2", "4", "more"]:
                            break
                        else:
                            error()
                    while objectAdd == 1:
                        lug_boot = input(f"What is the size of the luggage boot?(small, med, big): ")
                        if lug_boot in ["small", "med", "big"]:
                            break
                        else:
                            error()
                    while objectAdd == 1:
                        safety = input(f"How safe is the car?(low, med, high.): ")
                        if safety in ["low", "med", "high"]:
                            break
                        else:
                            error()
                object = simu(buying, maint, doors, persons, lug_boot, safety)
                simuList.append(object)
                objectAdd = int(input("Do you like to add another object? (1)yes (2)no: "))

                print("The following objects have been submitted to the list:")
                for j in simuList:
                    print(j.printObj())  
                print("The predicted class for each object is:")

                dfsimuList = pd.DataFrame([vars(obj) for obj in simuList])
                print(dfsimuList.columns.tolist())
                categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"] 
                
                # Updates the Preprocessor to consider the categorical data columns
                preprocessor2 = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(), categorical_features)
                    ]
                )

                # Apply preprocessing to the training set:
                preprocessed_df_simuList = preprocessor2.fit_transform(dfsimuList)
                predictions3 = knn.predict(preprocessed_df_simuList)
            else:
                print("The model has not been trained so far!")
                print("Please make sure you loaded and trained a data set")
                menu()

            

        case 5:
            end()

        case _:
            error()
            menu()
