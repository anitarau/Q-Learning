import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Define functions - will need to redefine for each value of alpha in loop below:

def placeholders():
    S_t0 = tf.placeholder(tf.float32, [None, state_size])
    a = tf.placeholder(tf.int32, [None, action_size])
    R = tf.placeholder(tf.float32, [None, 1])
    S_t1 = tf.placeholder(tf.float32, [None, state_size])
    
    return S_t0, R, a, S_t1

def mk_graph(lr, S_t0, a, R, S_t1):    # hidden , n_hidden=100, df=0.99
    
    hidden_layer = {'W': tf.Variable(tf.random_uniform([state_size, 100], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([100], 0, 0.01))}

    output_layer = {'W': tf.Variable(tf.random_uniform([100, action_size], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([action_size], 0, 0.01))}

    # hidden layer
    H_t0= tf.nn.relu(tf.matmul(S_t0,hidden_layer['W']) + hidden_layer['b'])
    H_t1 = tf.nn.relu(tf.matmul(S_t1,hidden_layer['W']) + hidden_layer['b'])

    # output layer
    Q = tf.matmul(H_t0,output_layer['W']) + output_layer['b']
    Q_old = tf.reshape(tf.gather_nd(Q, a), [-1, 1])
    Q_next = tf.matmul(H_t1, output_layer['W']) + output_layer['b']
    Q_max = tf.reshape(tf.reduce_max(Q_next, reduction_indices=[1]), [-1, 1])
    Q_target = R + discount * tf.multiply((1 + R), tf.stop_gradient(Q_max))
    
    loss = tf.reduce_mean(tf.square(Q_target - Q_old))/2
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    return  Q, Q_old, loss, Q_target, optimizer


""" function to test the current weights"""
def test_epoch(sess):   
    e_length = []
    e_return = []

    for episode_i in range(nTestEp):
        obs = env.reset()

        for t in range(300):
            Q_curr = sess.run(Q, feed_dict={S_t0: obs.reshape(1, 4)})
            act_curr = np.argmax(Q_curr)
            obs, rewa, done, info = env.step(act_curr)
            
            if done:                
                tot_ret = np.power(discount,t+1)  * -1
                e_return.append(tot_ret)
                e_length.append(t+1)
                break            
    return e_length, e_return

""" function to train model one epoch """
def train_epoch(sess):
    for b in range(num_batch):
        
        idx = np.random.randint(data.shape[0], size=batch_size)
        batch_data =  data[idx,:]                         

        a_batch = batch_data[:, 5].reshape(-1, 1)
        a_batch = np.append(np.arange(len(a_batch)).reshape(-1, 1), a_batch, axis=1) 
        batch_dict = {S_t0: batch_data[:, :4],
                      a: a_batch,
                      R: batch_data[:, 4].reshape(-1, 1),
                      S_t1: batch_data[:, 6:]
                      }
                      
        _,l = sess.run([optimizer,loss], feed_dict = batch_dict )
    return l
        



""" ----------------------- main ------------------------------ """   

# Set parameters
discount = 0.99
lr_dict = [0.00001]#, 0.0001, 0.001, 0.01, 0.1, 0.5]
batch_size = 100
nEp = 100
nTestEp = 100
data = np.load('data_q3_shifted.npy')
num_batch = int(data.shape[0] / batch_size )


env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

""" loop through all learning rates to compare results """
for lr in lr_dict:
    
    print("learning rate: ",lr)        
    epoch_mean_length, epoch_mean_return, epoch_loss = [],[],[]

    #env = gym.make('CartPole-v0')
    S_t0, R, a, S_t1 = placeholders()
    Q, Q_old, loss, Q_target, optimizer = mk_graph(lr,S_t0, a, R, S_t1)
    """ For each learning rate run nEP epochs. In each epoch first test the
        model with current weights, then train the model."""  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(nEp):
            print(epoch)
                  
            # test    
            e_length, e_return = test_epoch(sess)
            epoch_mean_length.append(np.mean(np.stack(e_length, axis=0)))        
            epoch_mean_return.append(np.mean(np.stack(e_return, axis=0)))
            
            # train
            epoch_loss.append(train_epoch(sess))
    # plot
    plt.plot(epoch_mean_length)
