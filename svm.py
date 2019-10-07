from sklearn.svm import SVC
import pandas as pd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Argument to train SVM")

parser.add_argument("--c", type=float, default=1.0, help="penalty parameter of the error term" )
parser.add_argument("--k", type=str, default="rbf",  choices=["rbf","linear", "poly", "sigmoid"], help="Kernel type")
parser.add_argument("--gamma", type=str, default="auto", choices=["auto","scale"], help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’")
parser.add_argument("--train-size", type=int, default=1000, help="The training set size")

args = parser.parse_args()


# load train and test data set
train_data = pd.read_csv("data/mnist_train.csv").values
test_data = pd.read_csv("data/mnist_test.csv").values

if args.train_size>train_data.shape[0]:
    train_size = train_data.shape[0]
else:
    train_size = args.train_size


def plot_confusion_matrix(confusion_matrix):
    labels = ["0","1","2","3","4","5","6","7","8","9"]
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("predicted labels")
    ax.set_ylabel("real labels")

    for i in range(10):
        for j in range(10):
            text = ax.text(j,i, confusion_matrix[i,j], ha="center", va="center", color="w")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


def train():
    # train 
    classifer = SVC(C=args.c, kernel=args.k, gamma=args.gamma, verbose=True)
    start_training_time = time.time()
    classifer.fit(train_data[np.arange(train_size)][:,np.arange(1, train_data.shape[1])], train_data[np.arange(train_size)][:,0])
    print("\n================================")
    print('Training time: {0:.4f} seconds'.format(time.time()-start_training_time))

    # build confusion matrix
    confusion_matrix = np.zeros((10,10))
    for i in range(test_data.shape[0]):
        predicted_label = classifer.predict([test_data[i][np.arange(1, test_data.shape[1])]])
        for j in np.arange(10):
            if test_data[i][0] == j:
                confusion_matrix[j][predicted_label]+=1
    print("Arguments used for training: ",args.__dict__)

    # Accuracy 
    acc = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print("Model Accuracy: {0:.4f} ".format(acc))

    # plot confusion matrix
    plot_confusion_matrix(confusion_matrix)
    

if __name__ == "__main__":
    train()

