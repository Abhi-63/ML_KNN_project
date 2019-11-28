from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt 
import time  
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
 



#spliting the data into number_of_flods_data 
def split_cross_validation(data, no_flods):
	data_split = list()
	data_copy = list(data)
	size = int(len(data) / no_flods)
	for _ in range(no_flods):
		temp = list()
		while len(temp) < size:
			idx = randrange(len(data_copy))         #copy the random data to temp 
			temp.append(data_copy.pop(idx))
		data_split.append(temp)
	return data_split
 

#finding the accuracy matrics
def accu_matric(actual, predict):
	crt = 0
	for j in range(len(actual)):
		if actual[j] == predict[j]:         #if actual is equal to predict we increment into 1
			crt =crt+1
	return crt / float(len(actual)) * 100.0   #calculating the accuracy percentage
 

#finding the k nearest neighbouring for train and test data
def k_nearest_neigh(train, test, no_neigh):
	predictions = list()
	for i in test:
		neigh = get_KNN(train, i , no_neigh)       #getting the neighbour points w.r.t test data set
		output_val = [i[-1] for i in neigh]         #getting values of the neigh
		predict = max(set(output_val), key=output_val.count) #optaining the maximum output_values
 
		predictions.append(predict)                #appending it to the prediction list
	return(predictions)


def dataset_min_max(data):
	min_max = list()
	for j in range(len(data)):
		col_val = [j[i] for j in data]  #iterating each column val wrt row
		val_min = min(col_val)        #min value among the column
		val_max = max(col_val)        #max value among the column
		min_max.append([val_min, val_max])
	return min_max

def e_d(x1 , x2):
    distance = 0.0
    for i in range(len(x1)-1):
        distance =distance+(x1[i] - x2[i])**2
    return sqrt(distance)

def manhattan_dist(x1 , x2):
    distance = 0.0
    for i in range(len(x1)-1):
        distance =distance+abs(x1[i] - x2[i])
    return distance

def get_KNN(train, test_a, no_neigh):
    distances = list()
    l=list()
  #  print(len(test_a))
   # print(len(train))
    for train_k in train:
        #print(train_k)
        #l.append(1)
        #dist = e_d(test_a , train_k)
        dist = manhattan_dist(test_a , train_k)
        distances.append((train_k, dist))         #appending the train_data and euclidean distance to the  list
    #print(len(l))
    distances.sort(key=lambda tup: tup[1])       #sorting the data into ascending order w.r.t euclidean distance
    neigh = list()
    for i in range(no_neigh):             #finding the neigh w.r.t euclidean distance
        neigh.append(distances[i][0])
    
    return neigh

#finding the normalize of the data
def normalization(data, min_max):
    for j in data:
        for i in range(len(j)):                 #finding the normalize of each row
            j[i] = (j[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])


def algo(data, algorithm, no_flods, *args):
    flods_data = split_cross_validation(data, no_flods)  #spliting the data into flods_data
    score = list()
    predicted = list()
    act = list()
    for part in flods_data:
        train_data = list(flods_data)              #spliting the data into train and test dataset 
        train_data.remove(part)
        train_data = sum(train_data, [])      
        test_data = list()
        for i in part:
            r_copy = list(i)
            test_data.append(r_copy)
            r_copy[-1] = None
        predict = algorithm(train_data, test_data, *args)   #predicting the neigh
        predicted.append(predict)
        #print(len(train_data))
        #print(len(test_data))
        actual = [i[-1] for i in part]
        act.append(actual)
        accuracy = accu_matric(actual, predict)
        score.append(accuracy)
        #print(act[0])
        #print(predicted[0])
        #print(len(act[0]))
        #print(len(predicted[0]))
    #results = confusion_matrix(act, predicted) 
    #print('Confusion Matrix :')
    #print(results) 
    #print('Report : ')
    #print(classification_report(act, predicted))

 

    return score
 
def avg(lst):
    return sum(lst)/len(lst)



#program starts from here

file_name = 'cat1.csv'                  #csv file 
data=list()
with open( file_name ,'r') as file:           #loading the csv file(readable)
#reading of csv file
    csv_read = reader(file)
    for k in csv_read:                          #iterating the csv file w.r.t each row
        if k:
            data.append(k)                    #appending the data into the list(in readable format)
        else:
            continue

r=len(data[0])
c=len(data)
#print(len(data[0]))  #row length
#print(len(data))     #coulum length

#converting the plain text into the math  form

for k in range(r-1): 
    for l in data:
        l[k] = float(l[k].strip())              #converting into the float from
    #print(data)
        
    #converting in the math form
r=len(data[0])-1

class_val = [i[r] for i in data]
unique = set(class_val)                      #eliminate the repeated values
temp = dict()                                   #initializing the dictinary type
for i, value in enumerate(unique):
    temp[value] = i
for j in data:
    j[r] = temp[j[r]]
    #print(data)
    
    
no_flods = 5  #divid the dataset into no_flods
no_neigh = 10 #K value


#print(data[0])

min_max = dataset_min_max(data)
#print(min_max)
#print(len(min_max))
normalization(data,min_max)
#print(data[0])
#print(len(normalize))

#t = list()
#a = list()
#j = 0

#k = [5,10,15,20,25,30]
#for i in k:
st=time.time()
scores = algo(data , k_nearest_neigh , no_flods , no_neigh)   #algorith to findthe  knn
en = time.time()
print('Scores: %s'% scores)
print("time", en-st)
#t.append(en-st)
acc = avg(scores)
print('Accuracy : ',acc,'%')
#a.append(acc)

#plt.plot(k,a)
#plt.title('Predict highest accuracy for "k value"')
#plt.xlabel('K value')
#plt.ylabel('Accuracy')

#plt.plot(k,t)
#plt.title('k value vs time using "Manhattan distance"')
#plt.xlabel('k value')
#plt.ylabel('Time')
