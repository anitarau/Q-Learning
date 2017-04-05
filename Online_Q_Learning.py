import gym
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt 

""" --------------------- functions ----------------------------------"""
def placeholders():
    S_t0 = tf.placeholder(tf.float32, [None, state_size])
    a = tf.placeholder(tf.int32, [None, 1])
    R = tf.placeholder(tf.float32, [None, 1])
    S_t1 = tf.placeholder(tf.float32, [None, state_size])
    done_tf = tf.placeholder(shape=[1],dtype=tf.bool) 
    return S_t0, R, a, S_t1, done_tf

def mk_graph(lr, S_t0, a, R, S_t1, done_tf):    # hidden , n_hidden=100, df=0.99
    
    hidden_layer = {'W': tf.Variable(tf.random_uniform([state_size, 100], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([100], 0, 0.01))}

    output_layer = {'W': tf.Variable(tf.random_uniform([100, action_size], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([action_size], 0, 0.01))}

    # hidden layer
    H_t0= tf.nn.relu(tf.matmul(S_t0,hidden_layer['W']) + hidden_layer['b'])
    H_t1 = tf.nn.relu(tf.matmul(S_t1,hidden_layer['W']) + hidden_layer['b'])

    # output layer
    Q = tf.matmul(H_t0,output_layer['W']) + output_layer['b']
    # Action value at old state under old Q
    a_onehot = tf.one_hot(a,action_size,dtype=tf.float32)
    Q_old  = tf.reduce_sum(tf.multiply(Q, a_onehot),axis=1)
    Q_next = tf.matmul(H_t1, output_layer['W']) + output_layer['b']
    # action value at new state (whith epsilon-greedy policy) under old Q
    Q_max = tf.reduce_max(Q_next,1)
    # target: bootstrap from new state
    Q_target = R + discount * tf.stop_gradient(Q_max)
    if done_tf is True: # can't bootstrap if termination state
        Q_target = R
    
    loss = tf.reduce_mean(tf.square(tf.sub(Q_target, Q_old)))/2
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    return  Q, Q_old, loss, Q_target, optimizer


# with probbaility epsion choose a random policy, where actions 0 and 1 equally likely
def rand_policy():
    u = np.random.uniform(-1,1)
    return math.ceil(u)

def test_epoch(sess):   
    e_length = []
    e_return = []

    for episode_i in range(n_test):
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
    
    
""" -------------------------- main ------------------------------"""

# define parameters
discount = 0.99
n_test = 100
epsilon = 0.05 # for epsilon-greedy policy
num_episodes = 1000
rep_freq = 20
lr = 0.001 # adjust this

# define placeholders for length, return and loss 
mean_lengths = []
mean_returns = []
mean_losses = []

# make agent
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0] #4
action_size = env.action_space.n #2

# define placeholders for old state, action, reward, new state, and indicator if episode terminated
S_t0, R, a, S_t1, done_tf = placeholders()
Q, Q_old, loss, Q_target, optimizer = mk_graph(lr,S_t0, a, R, S_t1, done_tf)

# start training 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        
    for cEp in range(num_episodes):
        
        """ --- train model one episode --- """
        obs = env.reset()
        sum_loss = 0

        for step in range(300):
            Q_= sess.run(Q, feed_dict={S_t0: obs.reshape(1, 4)})
            a_ = np.argmax(Q_)

            th = np.random.uniform()
            if th < epsilon:
                a_ = rand_policy()
            
            obs_ , R_, done, _ = env.step(a_)
            
            if done:
                R_ = -1
            else:
                R_ = 0

            dict_ = {S_t0: obs.reshape(-1,4),a: np.array([a_]).reshape(-1,1),
                     R: np.array([R_]).reshape(-1,1), S_t1: obs_.reshape(-1,4),
                     done_tf:np.reshape(done, [1])}

            _ , l = sess.run([optimizer, loss], feed_dict = dict_)
                   
            sum_loss += l
            obs = obs_
           
            if done: 
                break
                     
        """ --- test model after each cEp episodes --- """
        if (cEp % rep_freq)  == 0 :
            
            print(int(cEp*100/num_episodes),"%")
            mean_losses.append(sum_loss/step)
            lengths = []
            returns = []
            # run 100 episodes for testing and report
            for episode_i in range(n_test):
                obs_t = env.reset()

                for step_t in range(300):
                    Q_ep = sess.run(Q, feed_dict={S_t0: obs_t.reshape(1, 4)})
                    a_ep = np.argmax(Q_ep)
                    obs_t_, _ , done, _ = env.step(a_ep)

                    if done:
                        reward = -1
                        sum_rewards = reward * np.power(discount,step_t+1)
                        
                        lengths.append(step_t+1)
                        returns.append(sum_rewards)
                        break
                    obs_t = obs_t_

            mean_lengths.append(np.mean(np.stack(lengths, axis=0)))
            mean_returns.append(np.mean(np.stack(returns, axis=0)))

plt.plot(mean_lengths)          
mean_losses = np.reshape(np.stack(mean_losses,axis = 0),[-1,1])
mean_lengths = np.reshape(np.stack(mean_lengths,axis = 0),[-1,1])
mean_returns = np.reshape(np.stack(mean_returns,axis = 0),[-1,1])

print("mean length,max length,mean return, loss\n")
data = np.concatenate((mean_lengths,mean_returns,mean_losses),1)
print(data)    
    
