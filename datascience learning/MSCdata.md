---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Load theCalifornia Housing dataset
california_housing = fetch_california_housing(as_frame=True)
X= california_housing.data
y = california_housing.target

#Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Create a Linear Regression mode
linear_reg = LinearRegression()

#Train the model
linear_reg.fit(X_train, y_train)

#Make predictions on the test set
predictions = linear_reg.predict(X_test)

#Calculate mean squared error
mse= mean_squared_error(y_test, predictions)
print("Mean Squared Error is:",mse)
```

```python
#Scatterplot of predicted vs. actual values
import matplotlib.pyplot as plt
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

#line plot of actual and Predicted values
plt.plot(range(len(y_test)), y_test,label="Actual Values")
plt.plot(range(len(y_test)), predictions,label="Predicted Values")
plt.xlabel("Data Points")
plt.ylabel("Target Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()
```

```python

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load the Iris dataset 
iris = load_iris()
X = iris.data
y = iris.target

#split the dataset into training and testing sets
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a decision tress classifier 
dt_clas = sifier = DecisionTreeClassifier(random_state = 42)

# train the model 
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set 
predictions = dt_classifier.predict(X_test)

# calculate accuracy score 
accuracy = accuracy_score(y_test, predictions)
print("Decision Accuracy: ", accuracy)

# plot the decision tree ure_n
plt.figure(figsize=(10,6))
plot_tree(dt_classifier, feature_names = iris.feature_names, class_names = iris.target_names,filled = True)
plt.show().
```

```python
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with k = 3
knn_classifier = KNeighborsClassifier(n_neighbors = 3)

# Train the model
knn_classifier.fit(X_train, y_train)

#check 
predictions = knn_classifier.predict(X_test)

#Calculae accuracy score 
accuracy = accuracy_score(y_test, predictions)
print("KNN Accuracy:", accuracy)
```

```python
from sklearn.naive_bayes import GaussianNB

# create a Gaussian Naive Bayes Classifier
gnb = GaussianNB()

# train the model
gnb.fit(X_train, y_train)

# Make predictions on the test set 
predictions = gnb.predict(X_test)

#Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Naive Bayes Accuracy: ", accuracy)
```

```python
from sklearn import svm 

#create an svm(support vector machine)
svm_classifier = svm.SVC(kernel = 'linear')

#Train the model 
svm_classifier.fit(X_train, y_train)

#Make predictions on the test set 
predictions = svm_classifier.predict(X_test)

#calculate accuracy score 
accuracy = accuracy_score(y_test, predictions)
print("SVM accuracy:",accuracy)
```

unsupervised
- k-means clustering 
- principal component analysis 
- anomaly detection
- anomaly detection
- latent dirichlet allocation
- self organiziing maps 
- t-SNE 
- Hierarchical clustering
- association rule mining

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# generate synthetic data
X, _ = make_blobs(n_samples = 200 , centers = 3 , random_state = 42)

# Create a kmeans clustering model 
kmeans = KMeans(n_clusters=3, random_state= 42, n_init= 10 )

# fit the model to the data
kmeans.fit(X)

# Get the cluster labels for each data point 
labels = kmeans.labels_

# get the coordinates of the cluster centers
centers = kmeans.cluster_centers_

# visualize

plt.scatter(X[:, 0], X[:, 1], c = labels)
plt.scatter(centers[:, 0], centers[:, 1], marker = 'x', color = 'red')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree
```

```python
# Load the Iris dataset 
iris = load_iris()
X = iris.data
y = iris.target

#split the dataset into training and testing sets
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a decision tress classifier 
dt_classifier = DecisionTreeClassifier(random_state = 42)

# train the model 
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set 
predictions = dt_classifier.predict(X_test)

# calculate accuracy score 
accuracy = accuracy_score(y_test, predictions)
print("Decision Accuracy: ", accuracy)

# plot the decision tree ure_n
plt.figure(figsize=(10,6))
plot_tree(dt_classifier, feature_names = iris.feature_names, class_names = iris.target_names,filled = True)
plt.show()
```

```python
precision = precision_score(y_test, predictions,average='macro')
print("p score", precision)

#calculate recall
recall = recall_score(y_test, predictions, average='macro')
print("Recall Score:", recall)

# calculate f1-score
f1 = f1_score(y_test, predictions, average='macro')
print("F1 Score:", f1)

# calculate confusion matrix
confusion_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(confusion_mat)

# visuallize the confusion matrix
plt.figure(figsize=(10,6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names)
plt.yticks(tick_marks, iris.target_names)
plt.xlabel("predicted label:    ")
plt.ylabel("true label: ")
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        plt.text(j, i, confusion_mat[i, j], horizontalalignment='center', verticalalignment='center', color='white' if confusion_mat[i,j] > confusion_mat.max()/2 else 'black')
plt.show()

```

```python
# to remove underfitting or overfitting use cross validation k = 5
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Create a logistic regression model 
model = LogisticRegression()

# preform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the cross-validation scores
print("cross validation scores: ", cv_scores)
print("mean score:", cv_scores.mean())
```

```python
def summation(a, x, n, stop):
    sum = 0
    for r in range(stop):
        sum += term(a, x, r, n)
    return sum

def pi_function(n, r):
    product = 1
    if r == 0:
        return 1
    else:
        for k in range(r):
            product *= (n-k)
        return product

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    

def term(a, x, r, n):
    return ((a**(n-r) * x**r * pi_function(n, r)) / factorial(r))

def main():
    a = float(input("Enter the value of a: "))
    x = float(input("Enter the value of x: "))
    n = float(input("Enter the value of n: "))
    stop = int(input("Enter the value of stop: "))
    print("The value of(",a, "+", x, ")^", n, "is", summation(a, x, n, stop))

if __name__ == "__main__":
    main()

```

```python
def ab(stop,x):
    oldvalue=1 
    newvalue=1
    sum = 0
    for r in range (stop+1):
        sum += 1/(x**r)
        print(sum)

    print(sum-(1/(1-1/x)))

stop = int(input("Enter the value of stop: "))
x = float(input("Enter the value of x: "))
for i in range(stop):
    ab(i,x)

```

```python
def main():
    stop = int(input("Enter stop value"))
    print("2,3")
    a = 5
    
    for i in range(int(stop/4)):
        b = a + 2
        if (check_prime(a,b)):
            if a < stop and b < stop: 
                print(a,b) 
        a = b + 2


def check_prime(*args):
    i = 0
    no_of_no_prime = 0
    
    for arg in args:
        i+=1

    for arg in args:
        if (arg%2==0):
            no_of_prime += 1
            
        else: 
            for k in range(3,int((arg**.5)//1)+1,2):
                if (arg%k==0):
                    no_of_no_prime += 1
                    break

    if (no_of_no_prime == 0):
        return True
    else:
        return False
    
if __name__ == "__main__":
    main()
    

    


```

```python
def main():
    stop = int(input("Enter stop value"))
    print("2,3")
    a = 3
    b=0
    no_of_prime = 0

    for i in range(int(stop/2)-1):
        b = a + 2
        if (check_prime(a,b)):
            if a < stop and b < stop: 
                print("----->",a,b)
                no_of_prime += 1 
        a = b
    print(no_of_prime)


def is_prime(n):
    if (n%2==0):
        return 0
    else: 
        for k in range(3,int((n**.5)//1)+1,2):
            if (n%k==0):
                return 0
        return 1
    

def check_prime(*args):
    i = 0
    no_of_no_prime = 0
    
    for arg in args:
        i+=1

    for arg in args:
        if (arg%2==0):
            no_of_prime += 1
        elif is_prime(arg)==0:
            no_of_no_prime += 1

    if (no_of_no_prime == 0):
        return True
    else:
        return False
    
if __name__ == "__main__":
    main()
    

    


```

```python
import streamlit as st

st.title("My first streamlit app")
st.write("Welcome")
st.write("here, you can display text, data, and visualization")
```
