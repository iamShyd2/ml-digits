from math import gamma
from random import shuffle
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.4, shuffle=False)

clf.fit(x_train, y_train)

first_image = digits.images[0]

reshape = first_image.reshape(1, -1)

predicted = clf.predict(reshape)

print(predicted[0] == 0) # image is zero