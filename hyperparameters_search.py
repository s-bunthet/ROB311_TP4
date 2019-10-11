import subprocess
import numpy as np 
import re 
from progress.bar import Bar
from collections import OrderedDict
import time 

c_list = [0.01,0.1,1,10]
kernel_list = ["rbf","linear", "poly", "sigmoid"]
gamma_list = ["auto","scale"]

def get_accuracy(process_output):
    process_output = process_output.decode("utf-8")
    acc = re.findall("\d+\.\d+",process_output)[-1]
    return float(acc)


if __name__ == "__main__":
    ags_acc_dict = OrderedDict()
    start_time = time.time()
    bar = Bar('Searching Hyperparameters:', max=len(kernel_list)*len(gamma_list)*len(c_list))
    for c in c_list:
        for k in kernel_list:
            for gamma in gamma_list:
                process_output = subprocess.check_output(["python","svm.py", "--no-plot","--train-size", str(1000), "--k",str(k),"--gamma", str(gamma), "--c", str(c)])
                ags_acc_dict.update({"k: "+k+","+"gamma: "+gamma+","+"c: "+str(c): get_accuracy(process_output)})
                bar.next()
    bar.finish()
    ags_acc_dict = OrderedDict(sorted(ags_acc_dict.items(), key=lambda t: t[1]))
    print("Searching time: {0:.4f} seconds".format(time.time()-start_time))    
    print("The best Hyperparameters are: ", ags_acc_dict.popitem())

