import sys
import numpy as np 
import mnist
from mnist import MNIST
import pylab
from pylab import imshow, show, cm
import math
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics, pipeline
from sklearn.datasets import fetch_mldata
import utils
import argparse
import random
import matplotlib.pyplot as plt
import time
from tempfile import TemporaryFile  # save np array to file
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import randomized_svd
from sklearn.kernel_approximation import RBFSampler
import warnings
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''
PCA, Fourier Features, SGD, SDCA, mini-batch SDCA
'''
class RidgeRegression:
    def __init__(self,gamma, args):
        self.k=30000 # 1000
        self.n_comp=50
        self.method=args.method
        self.init_learn_rate = 0.001  # SGD init
        self.lamda = 0.00001  # regularizer for SGD should set to 0.00001  
        self.epoch = 20
        self.label_cnt = 10
        self.gamma = gamma
        self.sqrt_loss_list = list() # track lost at each epoch, include intial square error
        self.PJ_sqrt_loss_list = list() # track Polyak-Juditsky averaging weight loss, include intial error
        self.classification_loss_list=list() # track 1/0 loss at each epoch, ignore inital error
        self.PJ_classification_loss_list=list() # track 1/0 loss at each epoch

        self.test_sqrt_loss_list = list() # track lost at each epoch, include intial square error
        self.test_PJ_sqrt_loss_list = list() # track Polyak-Juditsky averaging weight loss, include intial error
        self.test_classification_loss_list=list() # track 1/0 loss at each epoch, ignore inital error
        self.test_PJ_classification_loss_list=list() # track 1/0 loss at each epoch

        # paras for SDCA
        self.sdca_g_loss_list = list() # track G loss
        self.SGD_training_time = list()

        # minibatch
        
        self.batch_training_class_loss = list()  # track batch loss
        # read
        self.read_dataset()
        # w for SGD
        # self.PJ_w_mat=np.zeros((self.label_cnt,self.k))  # Polyak-Juditsky averaging method.


    def read_dataset(self):
        # mndata = MNIST('./mnist')
        # self.train_img, self.train_label  = mndata.load_training()
        # self.test_img, self.test_label  = mndata.load_testing()

        print("fetching data")
        self.mnist = datasets.fetch_mldata('MNIST original',data_home='./mnist2')
        
       
        self.train_img =self.mnist.data[:60000]/255.
        self.train_label =self.mnist.target[:60000]
        self.test_img=self.mnist.data[60000:]/255.
        self.test_label =self.mnist.target[60000:]
        
        # # print("array shape",train_array.shape)
        print("load PCA transforming train and test data---")
        # self.train_pca= self.PCA_mnist(self.n_comp,self.train_img)
        #self.test_pca = self.PCA_mnist(self.n_comp,self.test_img)
        
        # save PCA results      
        #np.save('test_pca.npy', self.test_pca)
        #np.save('train_pca.npy', self.train_pca)
       
        # pca = PCA(n_components=self.n_comp)
        # self.train_pca = pca.fit_transform(self.train_img)
        # self.test_pca = pca.transform(self.test_img)
        #       # save PCA results      
        # np.save('test_pca.npy', self.test_pca)
        # np.save('train_pca.npy', self.train_pca)


        #self.test_pca = pca.fit(test_array).transform(test_array)
        #self.train_pca = pca.fit(train_array).transform(train_array)      

        # load PCA array
        self.test_pca = np.load('test_pca.npy')    
        self.train_pca = np.load('train_pca.npy')  

        # self.train_pca2 = np.load('train_pca.npy') 
        # self.test_pca2 = np.load('test_pca.npy')   
        self.train_count=self.train_pca.shape[0]  # 60000
        self.test_count = self.test_pca.shape[0]   # 10000    
        # print("self.test_count",self.test_count)
        # print("self.test_pca",self.test_pca[0])
        # print("self.test_pca2",self.test_pca2[0])

      
        # print("eigenvals",eigenvals[:10])
        print("shape of new_images",self.test_pca.shape)
        # print(" new_images",test_pca[0,:])
        self.rand_sample = np.sqrt(2 * self.gamma) * np.random.standard_normal(size=(self.n_comp,self.k))
        # self.rand_sample = np.random.standard_normal(size=(self.n_comp,self.k))
        self.random_offset =  np.random.uniform(0,2*np.pi,self.k)
        
        #--- random fourier ---------
        
        self.bandwidth = self.estimate_bandwidth(self.train_pca)
        print("kernel bandwidth", self.bandwidth)
        print("initial learning rate",self.init_learn_rate)
        print("gamma:",self.gamma)
        # t0=time.time()


    '''
    read mnist ref: https://martin-thoma.com/classify-mnist-with-pybrain/
    '''

    '''
    project 784 dimensions to n_comp=50
    input: array [image_count, dim]
    ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py 
    https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn 
    '''
    def PCA_mnist(self,n_comp,images):
        
        count, dim = images.shape
        # mean centered the data
      
        mean_ = np.mean(images,axis=0) 
        images -= mean_ #[60000,728]
        # get covariance matrix
        U, s, V = np.linalg.svd(images, full_matrices=False)  # V[728,728]
        
        components_ = V
        self.components_ = components_[:n_comp]
        S = np.diag(s[:n_comp])  #[50,50]
        U = U[:, :n_comp]   #[60000,50]
        new = np.dot(images, self.components_.T)
        return new


    '''
    Random Fourier Features to approximate RBF kernels
    input: image [1,50]
    return: transformed features h(x) of the size k=30000
    '''
    def random_fourier(self,image):
        # random sample [50,30000]  
        temp = np.dot(image,self.rand_sample)    # [1,50]*[50,30000]=[1,30000]
       
        temp = temp+self.random_offset
        # h = np.sin(temp/self.bandwidth)   # ref to homework
        h = np.cos(temp)* np.sqrt(2.)/np.sqrt(self.k)   
        # ref to https://arxiv.org/pdf/1506.02785.pdf
        return h


  

    '''
    randomly grab a few pairs of points and estimate the mean distance 
    between a random pair of points, multiplicatively cut it down
    by some factor (2)
    '''
    def estimate_bandwidth(self,images):
        # random sample 5 pairs of points
        mean_dist = 0.0
        num_pairs = 10
        for iter in range(0,num_pairs):
            # randomly sample 2 points from images
            rand_index = np.random.randint(images.shape[1], size=2)
            sampled =images[rand_index,:]
            # euclidian distance
            # fast computing of l2norm http://stackoverflow.com/questions/32141856/is-norm-equivalent-to-euclidean-distance
            dist = np.linalg.norm(sampled[0,:]-sampled[1,:])
            mean_dist+=dist
        return mean_dist/(num_pairs)

    '''
    turn unsigned int array to int
    '''
    def B_to_int(self):
        train_label=np.empty((len(self.train_label)))
        for i in range(0,len(self.train_label)):
            train_label[i]=self.train_label[i]
        return train_label

    def fit_SVM(self):
        classifier = svm.SVC(kernel='linear')
        # We learn the digits on the first half of the digits
                   # first permute data and associate their labels
        train_count = 60000
        seq = np.arange(self.train_count)
        np.random.shuffle(seq)  # random seq of 60000
        shuffled_images = self.train_pca[seq].reshape((self.train_count,self.n_comp))

        print("shuffled images shape",shuffled_images.shape)
        shuffled_labels = np.asarray(self.train_label)[seq] # a 1d array of 60000
        # shuffled_labels = self.get_y_true_3(shuffled_labels)
        # print("shuffled_labels[:n_samples / 2]",shuffled_labels[:,30000:30040])
        h_mat= np.empty((train_count,self.k))
        for i in range(0,train_count):
            h_mat[i,:] = np.asarray(self.random_fourier(shuffled_images[i,:]))
        #      h_mat[i,:]= self.rbf.fit_transform(shuffled_images[i,:])
        n_samples = train_count
        print("fitting")
    
        classifier.fit(h_mat[:n_samples / 4], shuffled_labels[:n_samples / 4].flatten())
        # classifier.fit(self.train_pca, self.train_label)

        h_test= np.empty((self.test_count,self.k))
        for i in range(0,self.test_count):
            # print("tranforming Fourier")
            h_test[i,:] = np.asarray(self.random_fourier(self.test_pca[i,:]))
      
        expected = self.test_label.flatten()

        # expected = self.test_label
        predicted = classifier.predict(h_test)
      
        # Now predict the value of the digit on the second half:

        # expected = shuffled_labels[:,n_samples/2:40000].flatten()
        # predicted = classifier.predict(h_mat[n_samples/2:40000])


        # mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
        #             solver='sgd', verbose=10, tol=1e-4, random_state=1,
        #             learning_rate_init=.1)

        # mlp.fit(X_train, y_train)
        # print("Training set score: %f" % mlp.score(X_train, y_train))
        # print("Test set score: %f" % mlp.score(X_test, y_test))
        print("Classification report for classifier %s:\n%s\n"
            % (classifier, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

 

    '''
    permute dataset, call linear classifier and SGD
    test at end of each epoch, images are of [60000,50]
    '''
    def run_PCA(self,method='SGD'):
        w_mat = np.zeros((self.label_cnt,self.k+1))   # 10,50/ 51 with intercept
        # # get intial square error

        # loop through epochs
        for ep in range(0,self.epoch):
           
            print("training epoch:",ep)
            # first permute data and associate their labels
            seq = np.arange(self.train_count)
            np.random.shuffle(seq)  # random seq of 60000
            shuffled_images = self.train_pca[seq].reshape((self.train_count,self.n_comp))
            shuffled_labels = np.asarray(self.train_label)[seq] # a 1d array of 60000
            shuffled_label_mat = self.get_y_true(shuffled_labels)  # get 10,60000 matrix
            # print("shuffled_label_mat",shuffled_label_mat[:,:5])
            # call train_linear_classifier
            if ep%10==0:
                self.init_learn_rate = self.init_learn_rate/5.0
            print("---begin training-----")
           
            new_w_mat,PJ_w_mat = self.train_PCA(w_mat,shuffled_images,shuffled_label_mat,shuffled_labels,ep,method='SGD')
            w_mat =new_w_mat
           
            
            self.plot_square_loss(ep, self.gamma)
            self.plot_zero_one_loss(ep, self.gamma)
            # do one test pass
            # print("-----begin testing------")
            # self.test_PCA(w_mat,PJ_w_mat,method='SGD')
        # plot training and test loss
        print("self.classification_loss_list",self.classification_loss_list)
        print("sqrt_loss_list",self.sqrt_loss_list)
        print("PJ_sqrt_loss_list",self.PJ_sqrt_loss_list)
        print("PJ_classification_loss",self.PJ_classification_loss_list)
        print("self.test_classification_loss_list.",self.test_classification_loss_list)
        print("test_sqrt_loss_list,",self.test_sqrt_loss_list)
        print("average training time:--",np.mean(self.SGD_training_time))


    '''
    train with SGD on fourier features
    '''
    def train_PCA(self,w_mat,train_imgs,train_labels_mat,train_labels,epoch_number,method='SGD'):
        # sample_cnt,dim = train_imgs.shape  #60000,50
        # weight matrix [10,30000] for 10 classifiers, 
        # each has a weight vector of size k=30,000    
        y_hat = np.empty((self.label_cnt,self.train_count))  # 10,60000
        PJ_y_hat = np.empty((self.label_cnt,self.train_count))  # 10,60000
        # PJ_w_mat=np.zeros((self.label_cnt,self.k)) ## Polyak-Juditsky averaging method.
        PJ_w_mat=np.zeros((self.label_cnt,self.k+1)) 
        sum_w_mat =np.zeros((self.label_cnt,self.k+1)) 
        right_class = 0.0
        sl = 0.0  # square loss
        pj_sl=0.0
        # iterate one pass  # SGD and update weight     
        t4 = time.time()
        
        for count in range(0,self.train_count):
            # get random_fourier for image and turn it into 30000 dims
            t3 = time.time()
            # h=train_imgs[count,:] # [,50]
            h = np.asarray(self.random_fourier(train_imgs[count,:]))   # [1,30000] turn to [30000,]
            # h = self.rbf.fit_transform(train_imgs[count,:])
            # expand dim to [1,50]        
            h = np.expand_dims(h,axis=0)
               # add intercept term:
            h = np.hstack((np.ones((h.shape[0], 1)), h)) # x_0 is always 1
            # print("w_mat",w_mat)
            t4= time.time()
            # print("complete random fourer takes----",t4-t3)
            y_hat[:,count]=np.dot(w_mat,h.T)[:,0]  #  product is [10,1]
            # temp=np.dot(w_mat,h.T)
            # PJ_y_hat[:,count]=np.dot(PJ_w_mat,h.T)[:,0]   # [10,30000] * [30000,10]
            # construct y_true to be [10,1]
            # y_true[train_labels[count],count]=1
            y_est= np.argmin(np.abs(1-y_hat[:,count]))  # true lable / index
            # pj_y_est = np.argmax(PJ_y_hat[:,count])

            hot_index = train_labels[count]  # true label such as 3
            if hot_index==y_est:                
                right_class+=1   
            # ----- weight update according to SGD ------#
            # w(t+1) = (1-2* eta(t)*lamda) * w(t) - 2 eta(t)(y_est-y_true)x_i
            #  w_mat [10,30000], ......[10,1]*[1,30000]
            # print ("iteration {}, epoch {}".format(count,epoch_number))
            w_mat = (1-2*self.init_learn_rate*self.lamda)*w_mat-2*self.init_learn_rate* np.dot(np.expand_dims(y_hat[:,count]-train_labels_mat[:,count],axis=1),h)
            # if epoch_number==self.epoch:
            sum_w_mat = sum_w_mat+w_mat   
            PJ_w_mat = sum_w_mat/float(count+1)
            PJ_y_hat[:,count]=np.dot(PJ_w_mat,h.T)[:,0]
            # l = train_labels_mat[:,0:count]-y_hat[:,0:count]  # [10,60000] or [10,10000]
            # l0 = train_labels_mat[:,count]-y_hat[:,count]
            lsl=train_labels_mat[hot_index,count]-y_hat[hot_index,count]
            sl +=lsl*lsl
            pjsls = train_labels_mat[hot_index,count]-PJ_y_hat[hot_index,count]
            pj_sl+=pjsls*pjsls

            if count % 100 == 0:
                # print("square loss, count",sl/(count+1), count)
                # print("average square loss, train label, y_hat, count",math.sqrt(np.sum(l**2)/(count+1)), train_labels_mat[:,count], y_hat[:,count], count)
                print("average 0-1 loss,count,epoch_number",1-right_class/float(count+1),count,epoch_number)
        # get square loss and 0/1 loss
        self.sqrt_loss_list.append(self.square_loss(train_labels_mat,y_hat,w_mat))
        self.PJ_sqrt_loss_list.append(self.square_loss(train_labels_mat,PJ_y_hat,PJ_w_mat))
        # self.classification_loss_list.append(self.zero_one_loss(train_labels,y_hat,self.train_count))
        self.PJ_classification_loss_list.append(self.zero_one_loss(train_labels,PJ_y_hat,self.train_count))
        # self.sqrt_loss_list.append(sl/(self.train_count))
        # self.PJ_sqrt_loss_list.append(pj_sl/(self.train_count))
        self.classification_loss_list.append(1-right_class/self.train_count)
        t5= time.time()
        self.SGD_training_time.append(t5-t4)
        print("-------time for training ----",t5-t4)
        print("-----begin testing------")
        self.test_PCA(w_mat,PJ_w_mat,method='SGD')
        return w_mat,PJ_w_mat



    def test_PCA(self,w_mat,PJ_w_mat,method='SGD'):
       # use self.test_pca to get 30000 features
        y_hat = np.empty((self.label_cnt,self.test_count))  # 10,10000
        PJ_y_hat = np.empty((self.label_cnt,self.test_count))  # 10,10000
        y_true = self.get_y_true(np.asarray(self.test_label))  # 10,10000
        right_class=0.0
        for count in range(0,self.test_count):
            # print("calculating random fourier at testing image %d"%count)
            h = self.random_fourier(self.test_pca[count,:])
           
            h = np.expand_dims(h,axis=0)
            h = np.hstack((np.ones((h.shape[0], 1)), h)) # x_0 is always 1
            y_hat[:,count]=np.dot(w_mat,h.T)[:,0]   #  product is [10,1]
            PJ_y_hat[:,count]=np.dot(PJ_w_mat,h.T)[:,0]    # [10,30000] * [30000,1]
            y_est = np.argmax(y_hat[:,count])
          
            if y_est == self.test_label[count]:
                right_class+=1.0
            if count % 100 == 0:
                # print("average square loss, train label, y_hat, count",math.sqrt(np.sum(l**2))/(count+1), train_labels_mat[:,count], y_hat[:,count], count)
                print("average 0-1 loss test---,count",1-right_class/float(count+1),count)
   
        # get loss
        self.test_sqrt_loss_list.append(self.square_loss(y_true,y_hat,w_mat))
        self.test_PJ_sqrt_loss_list.append(self.square_loss(y_true,PJ_y_hat,PJ_w_mat))
        self.test_classification_loss_list.append(self.zero_one_loss(self.test_label,y_hat,self.test_count))
        self.test_PJ_classification_loss_list.append(self.zero_one_loss(self.test_label,PJ_y_hat,self.test_count))
        self.plot_cm(y_hat,self.test_label)



    '''
    given a list of permuated training label [2,4,1,9....] in np.array [60000,]
    formulate 0/1 matrix [10,60000] or [10,10000] in testing cases
    '''
    def get_y_true(self,label_list):
        train_labels =np.zeros((self.label_cnt,label_list.shape[0]))
        for i in range(0,label_list.shape[0]):
            train_labels[int(label_list[i]),i]=1
        return train_labels


############################################################################
#########################################################################


    '''
    lw = sum(wx-y)^2 + lamda * ||w||
    RMSE = sum(wx-y)^2
    input : y_true: [10,60000],w_mat: [10,30000], could be PL_w_mat
    '''
    def square_loss(self,y_true,y_hat,w_mat):
        l = y_true-y_hat  # [10,60000] or [10,10000]
        ssq = np.sum(l**2) 
        loss = 1/float(y_true.shape[1]) * ssq
        print("squre loss----------",loss)
        return loss

    '''
    input: true labels in a list 
            y_hat in [10,60000] or [10,10000]
    '''
    def zero_one_loss(self,true_labels,y_hat,image_cnt):
        y_estimate = np.argmax(y_hat,axis=0)  # [60000, 1]
        # count of right estimate
        loss = len([i for i, j in zip(y_estimate, list(true_labels)) if i == j])
        # count = self.train_count if training
        print("classification loss---",1-(loss/float(image_cnt)))
        return 1-(loss/float(image_cnt))

    '''
    plot sum square loss of 10 classifiers. both training and test in one plot
    x axis is epoch number. Plot results from weights and averaged weights
    '''
    def plot_square_loss(self,ep,gamma,b=1,gamma2=None):
        fig = plt.figure(figsize=(10,8))
        x_range = np.array(range(0,ep+1))
        plt.title("square loss")
        plt.xlabel("epoch")
        plt.ylabel("sum loss")
        plt.grid()
        plt.plot(x_range,np.array(self.sqrt_loss_list),'o-',color='g',label="last point training loss")
        plt.plot(x_range,np.array(self.PJ_sqrt_loss_list),'o-',color='r',label="iterate averaging training loss")
        plt.plot(x_range,np.array(self.test_sqrt_loss_list),'o-',color='grey',label="last point test loss")
        plt.plot(x_range,np.array(self.test_PJ_sqrt_loss_list),'o-',color='orange',label="iterate averaging test loss")
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        filename = 'gamma_SDCA_square_loss_'+str(gamma2)+'_b_'+str(b)+'.png'
        fig.savefig(filename,bbox_inches="tight")
        plt.close()

    def plot_zero_one_loss(self,ep,gamma,b=1,gamma2=None):
        fig = plt.figure(figsize=(10,8))
        x_range = np.array(range(0,ep+1))
        plt.title("classification loss")
        plt.xlabel("epoch")
        plt.ylabel("sum loss")
        plt.grid()
        plt.plot(x_range,np.array(self.classification_loss_list),'o-',color='g',label="last point training loss")
        plt.plot(x_range,np.array(self.test_classification_loss_list),'o-',color='grey',label="last point test loss")
        plt.plot(x_range,np.array(self.PJ_classification_loss_list),'o-',color='r',label="iterate averaging training loss")
        plt.plot(x_range,np.array(self.test_PJ_classification_loss_list),'o-',color='orange',label="iterate averaging test loss")
        
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        filename = 'gamma_SDCA_classification_loss_'+str(gamma2)+'_b_'+str(b)+'.png'
        fig.savefig(filename,bbox_inches="tight")
        plt.close()


    '''
    plot 0-1 loss every batch
    '''
    def plot_batch_class_loss(self,batch_cnt,ep,gamma,b=1,gamma2=None):
        fig = plt.figure(figsize=(10,8))
        x_range = np.array(range(0,ep*self.train_count/b+batch_cnt+1))
        plt.title("batch classification loss")
        plt.xlabel("batch number")
        plt.ylabel("sum loss")
        plt.grid()
        plt.plot(x_range,np.array(self.batch_training_class_loss),'o-',color='g')
    
        filename = 'batch_classification_loss_'+str(gamma2)+'_b_'+str(b)+'.png'
        fig.savefig(filename,bbox_inches="tight")
        plt.close()


    '''
    plot sum square loss of 10 classifiers. both training and test in one plot
    x axis is epoch number. Plot results from weights and averaged weights
    '''
    def plot_g_loss(self,ep,gamma,b=1,gamma2=None):
        fig = plt.figure(figsize=(10,8))
        x_range = np.array(range(0,ep+1))
        plt.title("dual loss")
        plt.xlabel("epoch")
        plt.ylabel("dual loss")
        plt.grid()
        plt.plot(x_range,np.array(self.sdca_g_loss_list),'o-',color='g',label="training G loss")
 
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        filename = 'g_loss_'+str(gamma2)+'_b_'+str(b)+'.png'
        fig.savefig(filename,bbox_inches="tight")
        plt.close()

    '''
    confusion matrix
    input : y_true: [10,60000],w_mat: [10,30000], could be PL_w_mat

    '''
    def plot_cm(self,y_hat,labels):
        y_estimate = np.argmax(y_hat,axis=0)  # [60000, 1]

        print("Classification report for classifier")
        print(metrics.classification_report(labels, y_estimate))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, y_estimate))



    '''
    train 10 linear classifiers with l2 penalty and 
    update weights and alpha using (dual) coordinate ascent method
    compute G loss: a*=(I+XX.T/lamda)-1 * Y
    G = 1/2 a.T (I+XX.T / lamda) *a -Y.T *a 
      = 1/2 ||a||^2 + lamda/2 * ||w||^2  - Y.T *a
    update a:  delta_a = [ (yi-w*xi) -ai ]/(1+ (  ||xi||^2)/lamda  )
    update w: delta_w= 1/lamda * xi * delta_a
    a_mat : [10,60000]
    b: batch size
    '''
    def train_SDCA(self,b,a_mat,w_mat,train_imgs,train_labels_mat,train_labels,epoch_number,gamma2=None,method='SDCA'):
        # sample_cnt,dim = train_imgs.shape  #60000,50
        # weight matrix [10,30000] for 10 classifiers, 
        # each has a weight vector of size k=30,000    
        y_hat = np.empty((self.label_cnt,self.train_count))  # 10,60000
        PJ_y_hat = np.empty((self.label_cnt,self.train_count))  # 10,60000
        # PJ_w_mat=np.zeros((self.label_cnt,self.k)) ## Polyak-Juditsky averaging method.
        PJ_w_mat=np.zeros((self.label_cnt,self.k+1)) 
        sum_w_mat =np.zeros((self.label_cnt,self.k+1)) 
        right_class = 0.0
        sl = 0.0  # square loss
        pj_sl=0.0

        # gamma = 1/b if not specified
        if gamma2==None:
            gamma2 = 1/float(b)

       # 10,60000
        t4 = time.time()
        # loop trough batches
        for out_cnt in range(0,self.train_count/b):
            # loop within batch         
            delta_ai_mat = np.zeros((self.label_cnt,b))      # 10,100
            H = np.zeros((b,self.k+1))    # 100,30000   store transformed images in a batch
            for inner_cnt in range(0,b):
                count = out_cnt * b +inner_cnt    # reconstruct original index in 60000 samples 
                # get random_fourier for image and turn it into 30000 dims
                # h=train_imgs[count,:] # [,50]
                h = np.asarray(self.random_fourier(train_imgs[count,:]))   # [1,30000] turn to [30000,]
                # h = self.rbf.fit_transform(train_imgs[count,:])
                # expand dim to [1,50]
                h = np.expand_dims(h,axis=0)
                # add intercept term:
                h = np.hstack((np.ones((h.shape[0], 1)), h)) # x_0 is always 1
                H[inner_cnt,:] = h
                y_hat[:,count]=np.dot(w_mat,h.T)[:,0]  #  product is [10,1]
                PJ_y_hat[:,count]=np.dot(PJ_w_mat,h.T)[:,0]
                y_est= np.argmin(np.abs(1-y_hat[:,count]))  # true lable / index
                hot_index = train_labels[count]  # true label such as 3
                if hot_index==y_est:                
                    right_class+=1   

                # update amount [10,]  delta_ai [10,1]
                delta_ai = self.delta_ai(a_mat[:,count],y_hat[:,count],train_labels_mat[:,count],h)
                delta_ai_mat[:,inner_cnt]=delta_ai    # delta_ai_mat 100,10
                # update alpha_i,   [10,1]
                a_mat[:,count] = a_mat[:,count] + gamma2*delta_ai
                # delta_w[i,:] = delta_w[i,:] + 1/float(b)*1/float(self.lamda) * np.dot(h,delta_ai[i])


            # delta_ai_mat * H--->10,100 * 100,30000 = 10,30000
            w_mat = w_mat + gamma2*1/float(self.lamda) * np.dot(delta_ai_mat,H)
            # w_mat = (1-2*self.init_learn_rate*self.lamda)*w_mat-2*self.init_learn_rate* np.dot(np.expand_dims(y_hat[:,count]-train_labels_mat[:,count],axis=1),h)
            sum_w_mat = sum_w_mat+w_mat   
            PJ_w_mat = sum_w_mat/float(out_cnt+1)
            if out_cnt % 100 == 0:
                    # print("square loss, count",sl/(count+1), count)
                    # print("average square loss, train label, y_hat, count",math.sqrt(np.sum(l**2)/(count+1)), train_labels_mat[:,count], y_hat[:,count], count)
                print("average 0-1 loss,count,epoch_number",1-(right_class/(float(out_cnt+1)*b)),(out_cnt+1) * b-1,epoch_number)
            # self.batch_training_class_loss.append(1-(right_class/(float(out_cnt+1)*b)))
            # self.plot_batch_class_loss(out_cnt,epoch_number,self.gamma,b,gamma2)

        
        # get square loss and 0/1 loss
        self.sqrt_loss_list.append(self.square_loss(train_labels_mat,y_hat,w_mat))
        self.PJ_sqrt_loss_list.append(self.square_loss(train_labels_mat,PJ_y_hat,PJ_w_mat))
        # self.classification_loss_list.append(self.zero_one_loss(train_labels,y_hat,self.train_count))
        self.PJ_classification_loss_list.append(self.zero_one_loss(train_labels,PJ_y_hat,self.train_count))
        self.classification_loss_list.append(1-right_class/self.train_count)

        self.sdca_g_loss_list.append(self.dual_loss(train_labels_mat,w_mat,a_mat))
        t5= time.time()
        self.SGD_training_time.append(t5-t4)
        print("-------time for training ----",t5-t4)

        print("-----begin testing------")
        self.test_PCA(w_mat,PJ_w_mat,method='SGD')
        return w_mat,PJ_w_mat,a_mat



    '''
    permute dataset, call linear classifier and SDCA
    test at end of each epoch, images are of [60000,50]
    gamma2: mini-batch parameter
    b: batch size
    '''
    def run_SDCA(self,b,gamma2 = None,method='SGD'):
        w_mat = np.zeros((self.label_cnt,self.k+1))   # 10,50/ 51 with intercept
        a_mat = np.zeros((self.label_cnt,self.train_count))   # 10,50/ 51 with intercept
        # loop through epochs
        for ep in range(0,self.epoch):   
            print("training epoch:",ep)
            # first permute data and associate their labels
            seq = np.arange(self.train_count)
            np.random.shuffle(seq)  # random seq of 60000
            shuffled_images = self.train_pca[seq].reshape((self.train_count,self.n_comp))
            # print("shuffled images shape",shuffled_images.shape)
            shuffled_labels = np.asarray(self.train_label)[seq] # a 1d array of 60000
            # print(" shuffled_labels",shuffled_labels[:5])
            shuffled_label_mat = self.get_y_true(shuffled_labels)  # get 10,60000 matrix
          
            # no need of learning rate in SDCA case
            # if ep%5==0:
                # self.init_learn_rate = self.init_learn_rate/4.0
            print("---begin training-----")
           
            new_w_mat,PJ_w_mat,new_a_mat = self.train_SDCA(b,a_mat,w_mat,shuffled_images,shuffled_label_mat,shuffled_labels,ep,gamma2,method='SGD')
            w_mat =new_w_mat
            a_mat = new_a_mat
            self.plot_square_loss(ep, self.gamma,b,gamma2)
            self.plot_zero_one_loss(ep, self.gamma,b,gamma2)
            self.plot_g_loss(ep,self.gamma,b,gamma2)
            # do one test pass
            # print("-----begin testing------")
            # self.test_PCA(w_mat,PJ_w_mat,method='SGD')
        # plot training and test loss
        print("self.classification_loss_list",self.classification_loss_list)
        print("sqrt_loss_list",self.sqrt_loss_list)
        print("PJ_sqrt_loss_list",self.PJ_sqrt_loss_list)
        print("PJ_classification_loss",self.PJ_classification_loss_list)
        print("self.test_classification_loss_list.",self.test_classification_loss_list)
        print("test_sqrt_loss_list,",self.test_sqrt_loss_list)
        print("G-loss",self.sdca_g_loss_list)
        print("average training time:--",np.mean(self.SGD_training_time))
        # save result
        result_dict = {"classification_loss_list":self.classification_loss_list,
        "sqrt_loss_list":self.sqrt_loss_list,"PJ_sqrt_loss_list":self.PJ_sqrt_loss_list,
        "PJ_classification_loss":self.PJ_classification_loss_list,
        "test_classification_loss_list":self.test_classification_loss_list,
        "test_sqrt_loss_list":self.test_sqrt_loss_list,
        "sdca_g_loss_list":self.sdca_g_loss_list}
        pickle.dump(result_dict,open("SDCA_result.p",'wb'))

    

    '''
    compute G loss:
    a*=(I+XX.T/lamda)-1 * Y
    G = 1/2 a.T (I+XX.T / lamda) *a -Y.T *a 
      = 1/2 ||a||^2 + lamda/2 * ||w||^2  - Y.T *a
    input : y_true: [10,60000],w_mat: [10,30000], could be PL_w_mat
    a_mat: [10,60000], y_true: [10,60000]
    output: G [10,1]
  '''
    def dual_loss(self,y_true,w_mat,a_mat):
        G = 0.0
        a_norm_sum = 0.0
        w_norm_sum = 0.0
        a_times_y_sum = 0.0
        for i in range(0,self.label_cnt):
            # g = np.empty((self.label_cnt,1))
            g = 0.0
            # g = 1/2 * utils.l2_norm(a_mat[i,:])+ self.lamda/2.0 * utils.l2_norm(w_mat[i,:])-np.dot(a_mat[i,:],y_true[i,:].T) 
            a_norm = (1/2.0) * np.linalg.norm(a_mat[i,:])**2
            w_norm = self.lamda/2.0 * np.linalg.norm(w_mat[i,:])**2
            a_times_y = np.dot(a_mat[i,:],y_true[i,:].T)
            g = a_norm + w_norm - a_times_y 
            #print("dual loss debug: a_norm, w_norm, y*a: ", a_norm, w_norm, a_times_y)
        # print("y-true",y_true)
            G +=g
            a_norm_sum += a_norm
            w_norm_sum += w_norm
            a_times_y_sum += a_times_y
            #print("dual loss debug: individual g loss",g)    
        print("dual loss debug: dual loss, a_norm, w_norm, a_times_y",G,a_norm_sum,w_norm_sum,a_times_y_sum)
        return G


    '''
    delta_ai = [ (yi - w*hi) - ai  ] /  [1+ ||hi|| ^2 / lamda]
    input :  y_hat:[10,1],   y_true: [10,1]
    output: delta_ai [10,1] out of [10,60000]

    '''
    def delta_ai(self,ai,y_hat,y_true,hi):
        xi=hi.reshape(-1)   # flatten and get rid of (301,)
        # print("ai.shape",ai.shape)
        delta_a = (y_true - y_hat - ai) / (1+np.linalg.norm(xi)**2/float(self.lamda))
        # print("delta_ai",delta_a)
        return delta_a
 

 ####################################################################################





def main():
    args = get_args()
    gamma = 0.05  # for rbf 
    b = 100  # batch size
    gamma2 = 0.02
    RR = RidgeRegression(gamma,args)
    RR.run_SDCA(b,gamma2)


    # gamma2_list = np.linspace(0.001,0.03,30)
    # print gamma2_list
    
    # gamma2_list = [0.001,0.005,0.01,0.05,0.1,0.5,1]  # mini-batch para
    # # gamma2_list = [0.01,1]
    # min_loss_list=list()
    # for gamma2 in gamma2_list:
    #     print("gamma2",gamma2)
    # # gamma2 = 1000s
    #     RR = RidgeRegression(gamma,args)
    #     RR.run_SDCA(b,gamma2)
    #     min_loss = min(RR.classification_loss_list)
    #     print("min loss",min_loss)
    #     min_loss_list.append(min_loss)
    # print("min_loss_list",min_loss_list)


    # fig = plt.figure(figsize=(10,8))
    # x_range = gamma2_list
    # plt.title("gamma VS min classification loss")
    # plt.xlabel("gamma")
    # plt.ylabel("min loss")
    # plt.grid()
    # plt.plot(x_range,min_loss_list,'o-',color='g')
    #     # plt.plot(x_range,np.array(self.PJ_sqrt_loss_list),'o-',color='r',label="iterate averaging training loss")
    #     # plt.plot(x_range,np.array(self.test_sqrt_loss_list),'o-',color='grey',label="last point test loss")
    #     # plt.plot(x_range,np.array(self.test_PJ_sqrt_loss_list),'o-',color='orange',label="iterate averaging test loss")
    #     # plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    # filename = 'gamma_min_class_loss.png'
    # fig.savefig(filename,bbox_inches="tight")
    # plt.close()


    # gamma_list = [0.005]
    # loss_list = list()
    
    # for gamma in gamma_list:

    #     RR = RidgeRegression(gamma,args)
    #     # RR.run()
    #     RR.run_SDCA()
    #     loss_list.append(RR.test_classification_loss_list[-1])
    # print("loss.......",loss_list)
        
   
    # RR.fit_SVM()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method",action="store",type=str,
    default = 'SGD',metavar='M',help="input optimization methods: SGD, SDCA, minibatching")
    return parser.parse_args()



if __name__ == "__main__":
    main()