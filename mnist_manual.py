from sklearn import datasets
import numpy as np
import math
import operator
import matplotlib.pyplot as plt

def get_k_neighbors(X_train, y_train, point, k):
	distances = [] 
	neighbors = []

	for i in range(len(X_train)):
		distance = calc_distance(X_train[i], point) 
		distances.append((distance, y_train[i])) 
	distances.sort(key=operator.itemgetter(0))
	for i in range(k):
		neighbors.append(distances[i][1])
	return neighbors

def predict(X_train,y_train,p,k):
	neighbors_labels = get_k_neighbors(X_train, y_train, p, k) 
	return highest_votes(neighbors_labels)


def highest_votes(neighbors_labels):
	labels_count = [0,0,0,0,0,0,0,0,0,0]
	for label in neighbors_labels:
		labels_count[label] += 1
	max_count = max(labels_count) 
	return labels_count.index(max_count)

def calc_distance(p1,p2):
	dimension = len(p1)
	distance = 0
	for i in range(dimension):
		distance += (p1[i] - p2[i])**2
	return math.sqrt(distance)	
	
def accuracy_score(predicts, labels): 
	total = len(predicts)
	correct_count = 0 
	for i in range(total):
		if predicts[i] == labels[i]:
			correct_count +=1
	accuracy = correct_count/total	
	return accuracy

def main():
	digit = datasets.load_digits()
	digit_X = digit.data
	digit_y = digit.target

	randIndex = np.arange(digit_X.shape[0])
	np.random.shuffle(randIndex)
	digit_X = digit_X[randIndex]
	digit_y = digit_y[randIndex]

	# Slicing 
	X_train = digit_X[:1437,:] #1797-360
	X_test = digit_X[1437:,:] 
	y_train = digit_y[:1437] 
	y_test = digit_y[1437:] 

	k = 5
	y_predict = []
	for p in X_test:
		label = predict(X_train,y_train,p,k)
		y_predict.append(label)
	
	accuracy = accuracy_score(y_predict,y_test)
	accuracy = round(accuracy,3) * 100
	print("Accuracy: " + str(accuracy) + "%")

	# test value
	plt.gray()
	plt.imshow(X_test[0].reshape(8,8))
	print("Predict value: " + str(y_predict[0]))
	plt.show()

main()
