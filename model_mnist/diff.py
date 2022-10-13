import tensorflow as tf
import numpy as np
import csv
import argparse
import data
import matplotlib.pyplot as plt

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def diff(pred_1, pred_2):
    lst3 = [i for i, label in enumerate(pred_1) if label!=pred_2[i]]
    print(pred_1[:30])
    print(pred_2[:30])
    print(len(lst3))
    return len(lst3)

def True_False(y_pred_1, y_pred_2, y_test):

    count = 0
    for j, value in enumerate(y_test):
        if (y_pred_1[j]==value)&(y_pred_2[j]!=value):
            count += 1
    
    print(count)
    return count

def testing(y_pred1, y_pred2, y_test):
    diff = diff(y_pred1, y_pred2)
    TF = True_False(y_pred1, y_pred2, y_test)

    print('-----------------Testing Result-----------------')
    print('Diff: ', diff, 'Found: ', diff-TF, 'Error Detection rate: ', (diff-TF)/diff)

def analysis(y_pred1, y_pred2, y_test):
    diff = diff(y_pred1, y_pred2)

    acc_m1 = acc(y_test, y_pred1)
    acc_m1 = acc(y_test, y_pred2)
    approx_acc = 1 - (diff/len(y_test))

    print('-----------------Analysis Result-----------------')
    print('acc_m1: ', acc_m1, 'acc_m1: ', acc_m1, 'Diff #: ', diff, 'approx_acc: ', approx_acc)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--m1', type=str, required=True,
                help="Model1 name")
    parser.add_argument('--m2', type=str, required=True,
                help="Model2 name")
    parser.add_argument('--data', type=str, default='MNIST', choices=[
                    'MNIST', 'USPS', 'MNISTC'], help='Choose dataset (default: MNIST)')
    parser.add_argument('--method', type=str, default='Analysis', choices=[
                    'Testing', 'Analysis'], help='Choose method (default: Analysis)')

    args = parser.parse_args()
    model1_name = args.m1
    model2_name = args.m2

    if args.data == 'MNIST':
        d_select = data.KerasMnist()
    elif args.data == 'USPS':
        d_select = data.USPS()
    elif args.data == 'MNISTC':
        d_select = data.MNISTC()
    else:
        print('Choose the wrong dataset')

    
    x_test, y_test = d_select.test_data_28() 
    x_test_32, y_test_32 = d_select.test_data_32() 

    input_img = { # Experiments were conducted on the following types of models
        'lenet5' : x_test,
        'lenet5_2' : x_test,
        'lenet5_aug' : x_test,
        'lenet5_act' : x_test,
        'lenet5_random' : x_test,
        'lenet1' : x_test,
        'lenet1_act' : x_test,
        'lenet1_random' : x_test,
        'lenet1_2' : x_test,
        'lenet1_aug' : x_test,
        'resnet18' : x_test,
        'resnet18_2' : x_test,
        'resnet18_aug' : x_test,
        'resnet18_act' : x_test,
        'resnet18_random' : x_test,
        'resnet20' : x_test_32,
        'simplenet' : x_test_32,
        'simplenet_2' : x_test_32,
        'simplenet_aug' : x_test_32,
        'simplenet_act' : x_test_32,
        'simplenet_random' : x_test_32 ,
        'alexnet' : x_test_32,
        'alexnet_2' : x_test_32,
        'alexnet_aug' : x_test_32,
        'alexnet_act' : x_test_32,
        'alexnet_random' : x_test_32 
    }

    model1 = tf.keras.models.load_model("model/"+model1_name+"_mnist.h5")
    model2 = tf.keras.models.load_model("model/"+model2_name+"_mnist.h5")

    features1 = model1.predict(input_img[model1_name])
    features2 = model2.predict(input_img[model2_name]) 

    y_pred1 = features1.argmax(axis=1)
    y_pred2 = features2.argmax(axis=1)

    if args.data == 'Testing':
        testing(y_pred1, y_pred2, y_test)
    elif args.data == 'Analysis':
        analysis(y_pred1, y_pred2, y_test)
    else:
        print('Choose the wrong method')
    