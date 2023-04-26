import math
import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from torch.linalg import norm
from sklearn import datasets
from sklearn.preprocessing import normalize
from torchvision import datasets as torch_datasets
from torchvision import transforms
import scipy

###############################################

device = torch.device('cuda')
torch.manual_seed(42)

def get_data(data_name,partial_data,dp_num):
    # possible data_name: ['iris','bc','lfw_pairs','mnist','rcv1']

    if data_name == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X = X[y != 2]
        y = y[y != 2]
        y[y==0] = -1

        X /= X.max()  # Normalize X to speed-up convergence
    elif data_name == 'bc':
        bc = datasets.load_breast_cancer()
        X = bc.data
        y = bc.target
        if partial_data:
            X = X[0:dp_num]
            y = y[0:dp_num]
        y[y==0] = -1
        X = normalize(X,axis=0)  # Normalize X to speed-up convergence

    elif data_name == 'lfw_pairs':
        # train_set
        lfw_pairs = datasets.fetch_lfw_pairs(subset='train')
        X = lfw_pairs.data
        y = lfw_pairs.target
        names = lfw_pairs.target_names
        print("dataset names: {}".format(names))
        if partial_data:
            X = X[0:dp_num]
            y = y[0:dp_num]
        y[y==0] = -1
        
        # test_set
        lfw_pairs_test = datasets.fetch_lfw_pairs(subset='test')
        X_test = lfw_pairs_test.data
        y_test = lfw_pairs_test.target
        if partial_data:
            X_test = X_test[0:dp_num]
            y_test = y_test[0:dp_num]
        y_test[y_test==0] = -1


    elif data_name == 'mnist':
        train_data = torch_datasets.MNIST(
            root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
            train = True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
            download = True,
        )

        test_data = torch_datasets.MNIST(
            root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
            train = False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
            download = True,
        )

        loaders = {
            'train' : torch.utils.data.DataLoader(train_data,
                                                batch_size=60000,
                                                shuffle=True,
                                                num_workers=1),
            'test' : torch.utils.data.DataLoader(test_data,
                                                batch_size=10000,
                                                shuffle=True,
                                                num_workers=1)
        }

        X_train, y_train = next(iter(loaders['train']))
        X_test, y_test = next(iter(loaders['test']))
        X_train = torch.reshape(X_train,(-1,28*28))
        y_train[y_train%2==1] = 1
        y_train[y_train%2==0] = -1
        X_test = torch.reshape(X_test,(-1,28*28))
        y_test[y_test%2==1] = 1
        y_test[y_test%2==0] = -1


        if partial_data:
            X_train = X_train[0:dp_num]
            y_train = y_train[0:dp_num]
            X_test = X_test[0:dp_num]
            y_test = y_test[0:dp_num]

        X = X_train.to(device=device, dtype=torch.double)
        y = y_train.to(device=device, dtype=torch.double)
        X_test = X_test.to(device=device, dtype=torch.double)
        y_test = y_test.to(device=device, dtype=torch.double)

    elif data_name == 'rcv1':
        print('start reading data')
        X, y = datasets.load_svmlight_file('/home/buyun/datasets/rcv1_train.binary.bz2')


        if partial_data == False:
            X_test, y_test = datasets.load_svmlight_file('/home/buyun/datasets/rcv1_test.binary.bz2') # very large
        else:
            X_test = X[0:dp_num]
            y_test = y[0:dp_num]
            X = X[dp_num:]
            y = y[dp_num:]


        X = scipy.sparse.csr_matrix.toarray(X)
        X_test = scipy.sparse.csr_matrix.toarray(X_test)
        print('end reading data')

    else:
        print('please specify a legal data name')

    if data_name != 'mnist':
        X = torch.from_numpy(X).to(device=device, dtype=torch.double)
        y = torch.from_numpy(y).to(device=device, dtype=torch.double)
        [n,d] = X.shape
        X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.double)
        y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.double)
        [n_test,_] = X_test.shape
        
    else:
        n = X.shape[0]
        d = X.shape[1]
        n_test = X_test.shape[0]


    y = y.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    # d: size of data
    return [d,X,y,X_test,y_test,n,n_test]





# data_name = 'iris'
# data_name = 'bc' # breast cancer 
# data_name = 'lfw_pairs' # large dataset
data_name = 'mnist'
# data_name = 'rcv1' # document classification
[d,X,y,X_test,y_test,n,n_test] = get_data(data_name=data_name,partial_data=False,dp_num=10000)


def user_fn(X_struct,X,y,C):
    w = X_struct.w
    b = X_struct.b    
    zeta = X_struct.zeta

    # objective function
    f = 0.5*w.T@w + C * torch.sum(zeta)
    # inequality constraint 
    ci = pygransoStruct()
    constr1 = 1 - zeta - y*(X@w+b)
    constr2 = -zeta
    constr = torch.vstack((constr1,constr2)).to(device=device, dtype=torch.double)
    ci.c1 = torch.linalg.vector_norm(torch.clamp(constr, min=0),2) # l2

    # equality constraint
    ce = None

    return [f,ci,ce]


acc_tst = []
acc_tr = []
coefficient = []
# C_lst = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
# C_lst = [1e-6,1e-5,0.0001,0.001,0.01,0.1,1]

C_lst = [0.0001,0.001,0.01]

for C in C_lst:
    # variables and corresponding dimensions.
    var_in = {"w": [d,1], "b": [1,1], "zeta": [n,1]}
    comb_fn = lambda X_struct : user_fn(X_struct,X,y,C=C)



    opts = pygransoStruct()
    opts.torch_device = device
    opts.mu0 = min(1,1/C)
    opts.print_frequency = 10
    if C < 1e-2:
        opts.maxit = 1000
    else:
        opts.maxit = 5000
    opts.print_use_orange = False
    opts.print_ascii = True
    opts.quadprog_info_msg  = False
    opts.opt_tol = 1e-6
    opts.maxclocktime = 500
    opts.QPsolver = 'osqp'
    opts.limited_mem_size = 20
    opts.x0 =  torch.randn((d+1+n,1)).to(device=device, dtype=torch.double)
    opts.x0 = opts.x0/norm(opts.x0)
    # if C > 1:
    #     opts.x0[0:d] *= C

    soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)

    w = soln.final.x[0:d]
    b = soln.final.x[d:d+1]
    res = X@w+b
    predicted = torch.zeros(n,1).to(device=device, dtype=torch.double)
    predicted[res>=0] = 1
    predicted[res<0] = -1
    correct = (predicted == y).sum().item()
    acc = correct/n
    print("C = {}".format(C))
    print("train acc = {:.2f}%".format((100 * acc)))

    # obtain test acc
    res_test = X_test@w+b
    predict_test = torch.zeros(n_test,1).to(device=device, dtype=torch.double)
    predict_test[res_test>=0] = 1
    predict_test[res_test<0] = -1
    correct_test = (predict_test == y_test).sum().item()
    test_acc = correct_test/n_test
    print("test acc = {:.2f}%".format((100 * test_acc)))

    acc_tr.append(acc)
    acc_tst.append(test_acc)

import os
import matplotlib.pyplot as matplot

my_path = os.path.dirname(os.path.abspath(__file__))

matplot.subplots(figsize=(10, 5))
matplot.semilogx(C_lst, acc_tst,'-gD' ,color='red' , label="Testing Accuracy")
matplot.semilogx(C_lst, acc_tr,'-gD' , label="Training Accuracy")
#matplot.xticks(L,L)
matplot.grid(True)
matplot.xlabel("Cost Parameter C")
matplot.ylabel("Accuracy")
matplot.legend()
matplot.title('Accuracy versus the Cost Parameter C (log-scale)')
# matplot.show()



matplot.savefig(os.path.join(my_path, 'SVC_pygranso.png'))
print("test acc = {}".format(acc_tst))
print("train acc = {}".format(acc_tr))