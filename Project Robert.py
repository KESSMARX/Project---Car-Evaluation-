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

# Variables set up
menuEdit = 1

# While loop
while menuEdit == 1:
    
    # Set up of variables
    try:
        userChoice = int(input("Please type in the number of the choosen operation: "))
    
    except:
        print ("ERROR - wrong input!")
        userChoice = int(input("Please type in a number: "))

    match userChoice:
        case 1:
            # ---------------------------------------------------------------------------------------------
            # 1. LOADING
            def loadData():
                url = input("Insert the name for the dataset")
                df = pd.read_csv(url)
                classes = df["class"]
                print(classes.head(5))
            
            loadData()

            menuEdit = int(input("(1) continue or (2) cancel "))


        case 2:
            # ---------------------------------------------------------------------------------------------
            # 2. TRAINING 
            pass

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
            menuEdit = int(input("(1) continue or (2) cancel "))

# ---------------------------------------------------------------------------------------------