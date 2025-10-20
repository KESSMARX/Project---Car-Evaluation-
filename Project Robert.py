import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
knn = KNeighborsClassifier(n_neighbors=1)     # Creates a Knn Classifier Object with k=1
predictions = 0
features = 0
classes = 0
userattempt = 1

# Function set up
def loadData():
    global userattempt
    while userattempt == 1:
        try:
            url = input("Insert the name for the dataset: ")
            global successLoad # calling the global variable 
            successLoad = 1    # telling system the load was successfull
            df = pd.read_csv(url)
            print("The dataset was loaded sucessfully.")
            print(df.head(10))
            print(df.info())
            print(df.describe())
            print(df.shape)
            # Creates a dataframe using the drop method, which has two parameters:
            #   The first parameter tells which labels to remove (Columns Name) or 
            #   The second parameter tells whether to remove a row index or	
            #   a column name. axis=1 means we want to remove a column.
            global features # calling the global variable
            features = df.drop("class",axis=1)

            # Creates a dataframe from just one column:
            global classes # calling the global variable
            classes = df["class"]
            break
        except: 
            error()
            userattempt = int(input("\n(1) continue or (2) cancel "))
            if userattempt == 2:
                end()


def error():
    print ("\nERROR - wrong input!")


def menu():
    global menuEdit
    menuEdit = int(input("\n(1) continue or (2) cancel "))

     

def end():
    print ("\nThanks for your time and have a great day!\n")
    sys.exit()



# While loop
while menuEdit == 1:
    while userattempt == 1:
        try:
            userChoice = int(input("\nPlease type in the number of the choosen operation: "))
            break        
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

                userattempt = 1
                while userattempt == 1:
                    try:
                        userChoice = int(input("\nPlease type in the number of the choosen algorithm: "))
                        match userChoice:
                                            case 1:
                                                # Split the data into train/test sets
                                                features_train, features_test, classes_train, classes_test = train_test_split(
                                                features, classes, test_size=0.2, random_state=10
                                                )
                                        
                                                # Trains this Knn Classifier with the training set obtained previously:
                                                knn.fit(features_train, classes_train)

                                                predictions = knn.predict(features_test)

                                                print ("You have trained this set of data!")

                                            case 2:
                                                # Split the data into STRATIFIED train/test sets:
                                                strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
                                                features, classes, test_size=0.4, random_state=10, stratify=classes
                                                )
                                        
                                                # Trains this Knn Classifier with the training set obtained previously:
                                                knn.fit(strat_feat_train, strat_classes_train)

                                                predictions = knn.predict(strat_feat_test)
                                            
                                                print ("You have trained this stratified set of data!")  

                                            case _: 
                                                error()
                                                menu ()
                        break        
                   
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
# Code ending
if menuEdit == 2:
    end()



# ---------------------------------------------------------------------------------------------
# Code Testing
print (train_test_split)
