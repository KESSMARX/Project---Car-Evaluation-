import pandas as pd
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

# Function set up
def loadData():
    url = input("Insert the name for the dataset: ")
    print("The dataset was loaded sucessfully.")
    global successLoad # calling the global variable 
    successLoad = 1    # telling system the load was successfull
    df = pd.read_csv(url)
    print(df.head(10))
    print(df.info())
    print(df.describe())
    print(df.shape)

# While loop
while menuEdit == 1:
    
    try:
        userChoice = int(input("\nPlease type in the number of the choosen operation: "))
    
    except:
        print ("\nERROR - wrong input!")
        userChoice = int(input("Please type in a number: "))

    match userChoice:
        case 1:
            # ---------------------------------------------------------------------------------------------
            # 1. LOADING
            loadData()

            menuEdit = int(input("\n(1) continue or (2) cancel "))


        case 2:
            # ---------------------------------------------------------------------------------------------
            # 2. TRAINING 
            if successLoad == 1:
                print("**********************************************************")
                print("******************* Training Algorithms ******************")
                print("*                                                        *")
                print("* Please choose one of the following options:            *")
                print("*                                                        *")
                print("*(1) Split the data into train sets.                     *")
                print("*(2) Split the data into stratified train sets.          *")
                print("**********************************************************")

                userChoice = int(input("\nPlease type in the number of the choosen algorithm: "))

                match userChoice:
                    case 1:
                        # Split the data into train/test sets
                        features_train, features_test, classes_train, classes_test = train_test_split(
                        features, classes, test_size=0.2, random_state=10
                        )

                    case 2:
                        # Split the data into STRATIFIED train/test sets:
                        strat_feat_train, strat_feat_test, strat_classes_train, strat_classes_test = train_test_split(
                        features, classes, test_size=0.4, random_state=10, stratify=classes
                        )


            else:
                loadData()


            menuEdit = int(input("\n(1) continue or (2) cancel "))

        case 3:
            # ---------------------------------------------------------------------------------------------
            # 3. EVALUATION
            pass



        case 4:
            # ---------------------------------------------------------------------------------------------
            # 4. SIMULATION
            pass

        case _:
            print("Invalid input!")
            menuEdit = int(input("\n(1) continue or (2) cancel "))


    
# ---------------------------------------------------------------------------------------------
# Code ending
if menuEdit == 2:
    print ("\nThanks for your time and have a great day!\n")