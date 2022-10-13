import tensorflow as tf
import numpy as np
import csv
import argparse
import data

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
    print(len(lst3))
    return len(lst3)

def True_False(y_pred_1, y_pred_2, y_test):

    count = 0
    for j, value in enumerate(y_test):
        if (y_pred_1[j]==value)&(y_pred_2[j]!=value):
            count += 1
    
    print(count)
    return count

def load_model_10(model_name, x_test):
    model = tf.keras.models.load_model("model/"+model_name+"_cifar10.h5")
    features = model.predict(x_test)
    y_pred = features.argmax(axis=1)

    return y_pred

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
    parser.add_argument('--data', type=str, default='CIFAR10', choices=[
                    'CIFAR10', 'STL10', 'CIFAR10C'], help='Choose dataset (default: CIFAR10)')
    parser.add_argument('--method', type=str, default='Analysis', choices=[
                    'Testing', 'Analysis'], help='Choose method (default: Analysis)')

    args = parser.parse_args()
    model1_name = args.m1
    model2_name = args.m2

    if args.data == 'CIFAR10':
        d_select = data.KerasCifar10()
    elif args.data == 'STL10':
        d_select = data.STL10()
    elif args.data == 'CIFAR10C':
        d_select = data.Cifar10C()
    else:
        print('Choose the wrong dataset')

    
    x_test, y_test = d_select.test_data()

    y_pred1 = load_model_10(model1_name,x_test)
    y_pred2 = load_model_10(model2_name,x_test)

    if args.data == 'Testing':
        testing(y_pred1, y_pred2, y_test)
    elif args.data == 'Analysis':
        analysis(y_pred1, y_pred2, y_test)
    else:
        print('Choose the wrong method')
    