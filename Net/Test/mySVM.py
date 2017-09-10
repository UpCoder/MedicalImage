from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

def do_svc(traindata, trainlabel, testdata, testlabel):
    max_acc = 0.0
    for param_c in range(-20, 20, 1):
        for param_g in range(-20, 20, 1):
            svc = SVC(
                C=pow(2, param_c),
                gamma=pow(2, param_g)
            )
            svc.fit(traindata, trainlabel)
            predicted = svc.predict(testdata)
            acc = accuracy_score(predicted, testlabel)
            max_acc = max(acc, max_acc)
    print max_acc

if __name__ == '__main__':
    traindataart = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/train_art.npy'
    )
    trainlabelart = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/train_art_label.npy'
    )
    trainlabelpv = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/train_pv_label.npy'
    )
    traindatapv=  np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/train_pv.npy'
    )
    for index, label in enumerate(trainlabelart):
        if label != trainlabelpv[index]:
            print 'Error'
    traindata = np.concatenate([traindataart, traindatapv], axis=1)
    traindata = traindataart
    print np.shape(traindata)
    testdataart = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/val_art.npy'
    )
    testlabelart = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/val_art_label.npy'
    )
    testlabelpv = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/val_pv_label.npy'
    )
    testdatapv = np.load(
        '/home/give/PycharmProjects/MedicalImage/Net/data/val_pv.npy'
    )
    for index, label in enumerate(testlabelart):
        if label != testlabelpv[index]:
            print 'Error'
    testdata = np.concatenate([testdataart, testdatapv], axis=1)
    testdata = testdataart
    do_svc(traindata, trainlabelart, testdata, testlabelart)