# this file is referenced from online tutorials:
# 《基于LSTM和seq2seq的股票价格预测模型》https://howiemen.com/2018/08/03/%E5%9F%BA%E4%BA%8ELSTM%E5%92%8Cseq2seq%E7%9A%84%E8%82%A1%E7%A5%A8%E4%BB%B7%E6%A0%BC%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/

import pandas as pd
import numpy as np
import os 
os.chdir('/Users/liyuan/Desktop/SI699/codes')

class implement_seq2seq():
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self,window_size):
        data = pd.read_csv(self.data_path)
        predictor_names = ["price_usd"]
        training_features = np.asarray(data[predictor_names], dtype = "float32")
        kept_values = training_features[1000:]

        X = []
        Y = []
        for i in range(len(kept_values) - window_size * 2):#  x ；前window_size，y后window_size
            X.append(kept_values[i:i + window_size])
            Y.append(kept_values[i + window_size:i + window_size * 2])

        X = np.reshape(X,[-1,window_size,len(predictor_names)])
        Y = np.reshape(Y,[-1,window_size,len(predictor_names)])
        print(np.shape(X))
        return X, Y

    
    def generate_data(self, isTrain, batch_size):        
        # 40 pas values for encoder, 40 after for decoder's predictions.
        
        seq_length = 40   
        seq_length_test = 80

        global Y_train
        global X_train
        global X_test
        global Y_test
        # First load, with memoization:
        if len(Y_train) == 0:       
            X, Y= loadstock( window_size=seq_length)
            #X, Y = normalizestock(X, Y)

            # Split 80-20:
            X_train = X[:int(len(X) * 0.8)]
            Y_train = Y[:int(len(Y) * 0.8)]


        if len(Y_test) == 0:
            X, Y  = load_data( window_size=seq_length_test)
            #X, Y = normalizestock(X, Y)

            # Split 80-20:
            X_test = X[int(len(X) * 0.8):]
            Y_test = Y[int(len(Y) * 0.8):]

        if isTrain:
            return do_generate_x_y(X_train, Y_train, batch_size)
        else:
            return do_generate_x_y(X_test,  Y_test,  batch_size)
            

    def do_generate_x_y(X, Y, batch_size):
        assert X.shape == Y.shape, (X.shape, Y.shape)
        idxes = np.random.randint(X.shape[0], size=batch_size)
        X_out = np.array(X[idxes]).transpose((1, 0, 2))
        Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
        return X_out, Y_out
    

    def train_batch(batch_size):

        X, Y = generate_data(isTrain=True, batch_size=batch_size)
        feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
        feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})

        c =np.concatenate(( [np.zeros_like(Y[0])],Y[:-1]),axis = 0)

        feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})

        _, loss_t = sess.run([train_op, loss], feed_dict)
        return loss_t

    def test_batch(batch_size):
        X, Y = generate_data(isTrain=True, batch_size=batch_size)
        feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
        feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
        c =np.concatenate(( [np.zeros_like(Y[0])],Y[:-1]),axis = 0)#来预测最后一个序列
        feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})    
        output_lossv,reg_lossv,loss_t = sess.run([output_loss,reg_loss,loss], feed_dict)
        print("-----------------")    
        print(output_lossv,reg_lossv)
        return loss_t




