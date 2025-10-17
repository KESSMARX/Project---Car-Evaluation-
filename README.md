----- Task description -----

You will need to develop an application around the classification task.
Ideally you want to create a menu, which will allow the user to

1. Load the dataset
   The software must allow the user to type the dataset name, tell the user it was loaded correctly and then print:
   ● The first top 10 rows
   ● Some basic statistics
   
2. Train a classification model with the current version of the dataset
   The software must allow the user to select between at least 2 types of supervised machine learning algorithms to train the models.
   If no dataset was loaded, the software must ask the user to load a dataset first.
   
3. Evaluate and save the performance of the classification model
   The software must enable the user to choose whether or not they want to load a specific file for evaluation.
   In case the user does not provide the additional file, it performs the experiments using a suitable data partitioning strategy.
   After performing the experiments it will report suitable metrics for the user which will allow them to evaluate the performance of the classification model.
   After showing the metrics it will ask the user whether or not they want to save the results to a file.
   If the user says yes, the software asks for the file name and saves the results information there.
   
5. Simulate a real environment
   Simulate a real environment where the user can input a new unseen example without the class information and the system will return the result from using
   the classification model.
   If the user did not train any models yet, the software must tell the user that first they need to train a classification model.
   If the user trained methods, then it asks for the user to input information related to the attributes used for the problem and uses
   that information to ask the classification model for the correct response.
   The output of this method is the predicted class.


TO DO
- create a menu to load, train, evaluate and simulate a dataset
- 
