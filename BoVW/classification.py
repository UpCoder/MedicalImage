from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import numpy as np
from Tools import calculate_acc_error
import scipy.io as scio


class SVM:
    @staticmethod
    def do(train_data, train_label, test_data, test_label=None, adjust_parameters=False, C=1.0, gamma='auto'):
        train_data = np.array(train_data).squeeze()
        train_label = np.array(train_label).squeeze()
        test_data = np.array(test_data).squeeze()
        if test_label is not None:
            test_label = np.array(test_label).squeeze()
        if not adjust_parameters:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(train_data, train_label)
            predicts = clf.predict(test_data)
            acc = None
            if test_label is not None:
                acc = accuracy_score(test_label, predicts)
                # print acc
            return acc
        max_acc = 0.0
        max_predicted = None
        max_acc_train = 0.0
        target_c = None
        target_g = None
        for param_c in range(-20, 20, 1):
            for param_g in range(-20, 20, 1):
                clf = SVC(C=pow(2, param_c), gamma=pow(2, param_g))
                clf.fit(train_data, train_label)
                predicts = clf.predict(test_data)
                predicts_train = clf.predict(train_data)
                acc_train = accuracy_score(train_label, predicts_train)
                if acc_train > max_acc_train:
                    max_acc_train = acc_train
                acc = None
                if test_label is not None:
                    acc = accuracy_score(test_label, predicts)
                    # print acc
                    if acc > max_acc:
                        max_predicted = predicts
                        max_acc = acc
                        target_c = pow(2, param_c)
                        target_g = pow(2, param_g)
                    print 'training accuracy is ', acc_train, 'valication accuracy is ', acc
        print 'training max accuracy is ', max_acc_train, 'valication max accuracy is ', max_acc
        print 'target_c is ', target_c, ' target_g is ', target_g
        return max_predicted

class LinearSVM:
    @staticmethod
    def do(train_data, train_label, test_data, test_label=None):
        train_data = np.array(train_data).squeeze()
        train_label = np.array(train_label).squeeze()
        test_data = np.array(test_data).squeeze()
        if test_label is not None:
            test_label = np.array(test_label).squeeze()
        svm = LinearSVC()
        svm.fit(train_data, train_label)
        predicts = svm.predict(test_data)
        acc = None
        if test_label is not None:
            acc = accuracy_score(test_label, predicts)
            print acc
        return predicts, acc

if __name__ == '__main__':
    data = scio.loadmat('/home/give/PycharmProjects/MedicalImage/BoVW/data.mat')
    train_features = data['train_features']
    val_features = data['val_features']
    train_labels = data['train_labels']
    val_labels = data['val_labels']
    val_labels = np.squeeze(val_labels)
    print np.shape(train_features), np.shape(train_labels)
    predicted_label = SVM.do(train_features, train_labels, val_features, val_labels, adjust_parameters=True)

    np.save('./predicted_res.npy', predicted_label)
    # predicted_label = np.load('./predicted_res.npy')
    calculate_acc_error(predicted_label, val_labels)
