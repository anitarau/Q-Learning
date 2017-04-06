import gym
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt 

########## change directory if neccessary ###########
MODEL_FOLDER   = "C:/Users/Anita/Documents/UCL/AdvancedTopicsML/Assignment_3/models/q5/"
#####################################################

MODEL_FILENAME = MODEL_FOLDER+"q5_30u.ckpt" 



""" --------------------- functions ----------------------------------"""
def placeholders():
    S = tf.placeholder(tf.float32, [None, state_size])
    a = tf.placeholder(tf.int32, [None, 1])
    Q_target = tf.placeholder(tf.float32,[None, 1])
    return S, a, Q_target

def mk_graph(lr, S, a, Q_target):
    
    hidden_layer = {'W': tf.Variable(tf.random_uniform([state_size, n_units ], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([n_units ], 0, 0.01))}

    output_layer = {'W': tf.Variable(tf.random_uniform([n_units , action_size], 0, 0.01)),
                    'b': tf.Variable(tf.random_uniform([action_size], 0, 0.01))}

    # hidden layer
    H= tf.nn.relu(tf.matmul(S,hidden_layer['W']) + hidden_layer['b'])

    # output layer
    Q = tf.matmul(H,output_layer['W']) + output_layer['b']
    
    # action value at new state (whith epsilon-greedy policy) under old Q
    max_a = tf.argmax(Q,1)
    #a_onehot = tf.one_hot(max_a,action_size,dtype=tf.float32)
    Q_max  = tf.reduce_max(Q,1)
    # action value at new state (whith epsilon-greedy policy) under old Q
   
    loss = tf.reduce_mean(tf.square(tf.sub(tf.stop_gradient(Q_target), Q_max)))/2
    optimize = tf.train.AdamOptimizer(lr).minimize(loss)
    return  Q, Q_max, max_a, loss, optimize


# with probbaility epsion choose a random policy, where actions 0 and 1 equally likely
def rand_policy():
    u = np.random.uniform(-1,1)
    return math.ceil(u)

    
def plot_means():
    plt.figure(1)   
    plt.plot(mean_lengths) 
    plt.savefig('q5_30u_mean_lengths.png')    
    plt.figure(2)    
    plt.plot(mean_returns) 
    plt.savefig('q5_30u_mean_returns.png')
    plt.figure(3)   
    plt.plot(mean_losses) 
    plt.savefig('q5_30u_mean_losses.png') 
    #plt.show()

    
    
""" -------------------------- main ------------------------------"""

# define parameters
discount = 0.99
n_test = 100
epsilon = 0.05 # for epsilon-greedy policy
num_episodes = 2000
lr = 0.001 # adjust this
rep_freq = 20
n_units = 30

# define placeholders for length, return and loss 
mean_lengths, mean_returns, mean_losses = [],[],[]

# make agent
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0] #4
action_size = env.action_space.n #2

# define placeholders for old state, action, reward, new state, and indicator if episode terminated
S, a, Q_target = placeholders()
Q, Q_max, max_a, loss, optimize = mk_graph(lr, S, a, Q_target)

# start training 
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        
    for cEp in range(num_episodes):
        
        """ --- train model one episode --- """
        obs = env.reset()
        sum_loss = 0

        for step in range(300):
            max_a_ = sess.run(max_a, feed_dict={S: obs.reshape(1, 4)})
            max_a_ = np.asscalar(max_a_)

            th = np.random.uniform()
            if th < epsilon:
                max_a_ = rand_policy()
            
            obs_ , R, done, _ = env.step(max_a_)
            
            if done:
                R = -1
            else:
                R = 0
                
            Q_max_ = sess.run([Q_max], feed_dict = {S: obs_.reshape(-1,4)})
            new_target =  R + (1+R) * discount * np.asmatrix(Q_max_)
            dict_ = {S: obs.reshape(-1,4),a: np.array([max_a_]).reshape(-1,1),
                     Q_target: new_target.reshape(-1,1)}

            _ , l = sess.run([optimize, loss], feed_dict = dict_)
                   
            sum_loss += l
            obs = obs_
           
            if done: 
                break
                     
        """ --- test model after each episode --- """
                    
        if (cEp % rep_freq)  == 0 :
            if (cEp % 200) == 0:
                print(int(cEp*100/num_episodes),"%", end='\r') #print progress
            mean_losses.append(sum_loss/step)
            lengths = []
            returns = []
            # run 100 episodes for testing and report
            for _ in range(n_test):
                obs_t = env.reset()
                for step_t in range(300):
                    a_ep = sess.run(max_a, feed_dict={S: obs_t.reshape(1, 4)})
                    obs_t, _ , done, _ = env.step(np.asscalar(a_ep))
    
                    if done:
                        sum_rewards = -1 * np.power(discount,step_t+1)                    
                        lengths.append(step_t+1)
                        returns.append(sum_rewards)
                        break
    
            mean_lengths.append(np.mean(np.stack(lengths, axis=0)))
            mean_returns.append(np.mean(np.stack(returns, axis=0)))
    print('Training finished')        
    saver.save(sess, MODEL_FILENAME)
    print('Model saved') 
plot_means()



