from sklearn.svm import SVC
import pandas as pd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Argument to train SVM")

parser.add_argument("--c", type=float, default=10.0, help="penalty parameter of the error term" )
parser.add_argument("--k", type=str, default="rbf",  choices=["rbf","linear", "poly", "sigmoid"], help="Kernel type")
parser.add_argument("--gamma", type=str, default="scale", choices=["auto","scale"], help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’")
parser.add_argument("--train-size", type=int, default=-1, help="The training set size. '-1' mean that we use all the dataset ")
parser.add_argument("--seed", type=int, default=1, help="The random seed.")
parser.add_argument("--no-plot", default=False, action="store_true", help="Don't plot the confusion matrix")
parser.add_argument("--vb", default=False,action="store_true",help="Turn on verbose")

args = parser.parse_args()


# load train and test data set
train_data = pd.read_csv("data/mnist_train.csv").values
test_data = pd.read_csv("data/mnist_test.csv").values

# we shuffle the train and also test data since we might not use the whole data for training and testing. 
# Shuffling make sure that the subset of the train or test data contain data with all labels.
np.random.seed(args.seed)
np.random.shuffle(train_data) 
np.random.shuffle(test_data)    
 


if args.train_size>train_data.shape[0] or args.train_size ==-1:
    train_size = train_data.shape[0]
else:
    train_size = args.train_size

test_size = int(train_size/4) # test with 20% of the size of train data 
if test_size > test_data.shape[0]:
    test_size = test_data.shape[0]


def plot_confusion_matrix(confusion_matrix):
    """
    Plot a confusion matrix.
    :param confusion_matrix:
    :return:
    """
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
    """
    Train SVC (C-Support Vector Classification)
    :return:
    """
    # train 
    classifer = SVC(C=args.c, kernel=args.k, gamma=args.gamma, verbose=args.vb)
    start_training_time = time.time()
    classifer.fit(train_data[np.arange(train_size)][:,np.arange(1, train_data.shape[1])], train_data[np.arange(train_size)][:,0])
    print("\n================================")
    print('Training time: {0:.4f} seconds'.format(time.time()-start_training_time))

    # build confusion matrix
    start_prediction_time = time.time()
    confusion_matrix = np.zeros((10,10))
    for i in range(test_size):
        predicted_label = classifer.predict([test_data[i][np.arange(1, test_data.shape[1])]])
        for j in np.arange(10):
            if test_data[i][0] == j:
                confusion_matrix[j][predicted_label]+=1
    print("Arguments used for training: ",args.__dict__)
    print('Prediction time: {0:.4f} seconds'.format(time.time()-start_prediction_time))
    # Accuracy 
    acc = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print("Model Accuracy: {0:.4f} ".format(acc))

    # plot confusion matrix
    if not args.no_plot:
        plot_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    train() 

