#------------------------------------------------------------------------
# AUTHOR: Josephine Nguyen
# FILENAME: decision_tree.py
# SPECIFICATION: This program reads 3 different training datasets (each with increasing instances of data), constructs a decision tree, and then runs it 10 times with a given test dataset. Out of the 10 times, the most minimal/best accuracy is chosen as the final accuracy of the dataset. All 3 final accurracies are ultimately stored and outputted.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 60 min
# #-----------------------------------------------------------*/
#
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

#Purpose:   Used to transfrom attributes for X[] and dsTest[], may or may not include the last classfication attribute:
# Age:                    (Young = 1), (Presbyopic = 2), (Prepresbyopic = 3)
# Spectacle Prescription: (Myope = 1), (Hypermetrope = 2)
# Astigmatism:            (No = 1), (Yes = 2)
# Tear Production Rate:   (Reduced = 1), (Normal = 2)
# Recommended Lenses:  (Yes = 1), (No = 2)
def parseAttributes(dataList, includeRecLenses):
    twoDArray = []

    for attribute in dataList:
        currRow = []
        for valueIndex in range(len(attribute)):
            value = attribute[valueIndex]

            #Check Age values:
            if valueIndex == 0:
                if value == "Young":
                    currRow.append(1)
                elif value == "Presbyopic":
                    currRow.append(2)
                elif value == "Prepresbyopic":
                    currRow.append(3)
                else:
                    print("something went wrong")

            # Check Spectacle values:
            elif valueIndex == 1:
                if value == "Myope":
                    currRow.append(1)
                elif value == "Hypermetrope":
                    currRow.append(2)
                else:
                    print("something went wrong")

            # Check Astigmatism values:
            elif valueIndex == 2:
                if value == "No":
                    currRow.append(1)
                elif value == "Yes":
                    currRow.append(2)
                else:
                    print("something went wrong")

            # Check Tear values:
            elif valueIndex == 3:
                if value == "Reduced":
                    currRow.append(1)
                elif value == "Normal":
                    currRow.append(2)
                else:
                    print("something went wrong")

            if includeRecLenses is True and valueIndex == 4:
                if value == "Yes":
                    currRow.append(1)
                elif value == "No":
                    currRow.append(2)
                else:
                    print("something went wrong")
        twoDArray.append(currRow)
    return twoDArray


#Start Problem Answer here:
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
final_accuracies = []  #to store all final accuracies of each dataset after running each 10 times

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original training features to numbers and add to the 4D array X.:
    X = parseAttributes(dbTraining, False)

    #transform the original training classes to numbers and add to the vector Y.
    # Recommended Lenses:  (Yes = 1), (No = 2)
    for attribute in dbTraining:
        value = attribute[4]            #get only the value for the Rec Lenses attribute
        # Check Rec Lenses values:
        if value == "Yes":
            Y.append(1)
        elif value == "No":
            Y.append(2)
        else:
            print("something went wrong")


    # #loop your training and test tasks 10 times here
    min_accuracy = 1            #to track most min accuracy as we run
    for i in range(10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)


        #Read contact_lens_test.csv to get test data into dbTest:
        dbTest = []
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)

            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        # Parse dbTest into values just like dbTraining, DO include the rec lenses attribute:
        dbTestTransformed = parseAttributes(dbTest, True)


        #Make a predictionfor every test data instance + compare it to the actual answer (Rec Lenses attribute)
        true_predictions = 0            #to track how many predictions were wrong/right
        false_predictions = 0
        for data in dbTestTransformed:
            onlyAttributes = [[data[0], data[1], data[2], data[3]]]

            #Use the decision tree to make the class prediction.
            class_predicted = clf.predict(onlyAttributes)[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if data[4] == class_predicted:
                true_predictions = true_predictions+1
            else:
                false_predictions = false_predictions+1

        #Calculate and compare to current min accuracy:
        curr_accuracy = true_predictions/(true_predictions+false_predictions)
        if curr_accuracy < min_accuracy:
            min_accuracy = curr_accuracy

    #Noq that we've run the test 10 times, print the lowest accuracy of this model (training and test set) and save it.
    final_accuracies.append(min_accuracy)
    print("Final accuracy when training on " + ds +":  " + str(min_accuracy))
