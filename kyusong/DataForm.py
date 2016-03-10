import cPickle

class DataForm:
    def load(self):
        X_test = cPickle.load(open("test_X_0.cPickle"))
        Y_test = cPickle.load(open("test_y_0.cPickle"))
        X_train = cPickle.load(open("train_X_0.cPickle"))
        Y_train = cPickle.load(open("train_y_0.cPickle"))

        X_test = self.chageX(X_test)
        X_train = self.chageX(X_train)
        Y_test = self.chageY(Y_test)
        Y_train = self.chageY(Y_train)
        return (X_train, Y_train),(X_test, Y_test)
    
    def chageX(self,X_test):
        List = []
        for X in X_test:
            ListX = []
            for x in X:
                ListX.append(int(x))
            List.append(ListX)
        return List

    def chageY(self,X_test):
        List = []
        for x in X_test:
            List.append(int(x))
        return List
