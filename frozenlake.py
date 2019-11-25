import numpy as np
import gym
from gym import wrappers
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import random
import pandas as pd
import time

random_map = generate_random_map(size=20, p=0.8)

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        #print "Reward received for each action"
        #print reward
        total_reward += (gamma ** step_idx * reward)
        #print "Total reward inside whileloop"
        #print total_reward
        step_idx += 1
        if done:
            break
    #print "My Total Reward in run_episode function"
    #print total_reward
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    value_list = []
    max_iterations = 100000
    #max_iterations = 10
    eps = 1e-10
    state_values = np.zeros((max_iterations,env.nS))
    for i in range(max_iterations):
        start = time.time()
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
            state_values[i][s] = v[s]
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
        value_list.append(v)
        print(time.time()-start)
    value_array = np.array(value_list)
    #print "Value array shape"
    #print value_array.shape
    #print value_array[10]
    x = np.arange(16)
    plt.plot(x,value_array[10])
    plt.plot(x,value_array[100])
    plt.plot(x,value_array[200])
    plt.plot(x,value_array[500])
    plt.plot(x,value_array[1000])
    plt.plot(x,value_array[2000])
    plt.legend("10,100,200,500,1000,2000",loc=center)
    #plt.show()
    #print value_array[50]
    #print value_array[100]
    #print value_array[150]
    #print value_array[200]
    #print value_array[250]
    #plt.figure()
    #plt.plot([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],value_array[10])
    #plt.plot(100,value_array[100])
    #plt.plot(150,value_array[150])
    #plt.title("Values of different states")
    #plt.show()
    #print "Value functions accumulated State Values"
    #print state_values
    return v




def display_iteration(P,V,env,alg):
    #env = gym.make(env_name,desc=random_map)
    nb_states = env.observation_space.n
    #env.seed(0)
    dim = int(np.sqrt(nb_states))
    print "My dimension",dim
    policy = P.reshape(dim,dim)
    vi = V.reshape(dim,dim)
    plt.figure(figsize=(3,3))
    #plt.imshow(V.reshape(dim,dim), cmap='gray', interpolation='none', clim=(0,1))
    plt.imshow(V.reshape(dim,dim), interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(dim)-.5)
    ax.set_yticks(np.arange(dim)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:dim, 0:dim]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    for y in range(dim):
        for x in range(dim):
            a = policy[y,x]
            u,v = a2uv[a]
            plt.arrow(x,y,u*.3,-v*.3,color='m',head_width=0.1,head_length=0.1)
            plt.text(x,y,str(env.unwrapped.desc[y,x].item().decode()),color='g',size=12,verticalalignment='center',horizontalalignment='center',fontweight='bold')
    plt.grid(color='b',lw=2,ls='-')
    plt.title(alg)
    plt.savefig(alg+'.png')
    plt.show()

def display_qiteration(P,env,alg):
    #env = gym.make(env_name,desc=random_map)
    nb_states = env.observation_space.n
    #env.seed(0)
    dim = int(np.sqrt(nb_states))
    print "My dimension",dim
    policy = P.reshape(dim,dim)
    plt.figure(figsize=(3,3))
    plt.imshow(P.reshape(dim,dim), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(dim)-.5)
    ax.set_yticks(np.arange(dim)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:dim, 0:dim]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    for y in range(dim):
        for x in range(dim):
            a = policy[y,x]
            u,v = a2uv[a]
            plt.arrow(x,y,u*.3,-v*.3,color='m',head_width=0.1,head_length=0.1)
            plt.text(x,y,str(env.unwrapped.desc[y,x].item().decode()),color='g',size=12,verticalalignment='center',horizontalalignment='center',fontweight='bold')
    plt.grid(color='b',lw=2,ls='-')
    plt.title(alg)
    plt.savefig(alg+'.png')
    plt.show()

def qlearning(lr,y,num_episodes,env):
    print "Q Learning"
    # initialize Q-Table
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    # create lists to contain total rewards and steps per episode

    rList = []
    sList = []

    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        sList=[]
    # The Q-Table learning algorithm
    while not d and j<250:
        j+=1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(5./(i+1)))

        # Get new state and reward from environment
        s1,r,d,_ = env.step(a)

        # Get negative reward every step
        if r==0 :
            r=-0.001

    # Q-Learning
        Q[s,a]= Q[s,a]+lr*(r+y* np.max(Q[s1,:])-Q[s,a])
        s=s1
        rAll=rAll+r
        sList.append(s)

    rList.append(rAll)
    if r==1 :
        print(sList)
    print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, j,rAll ,np.mean(rList)))
    q_policy = Q.argmax(axis=1)
    return Q,q_policy

def gammas_training(env,method,gammas = np.arange(0,1,0.1)):
    df = pd.DataFrame(columns=['gamma','state','value'])
    for gamma in gammas:
        if method == 'value_iteration':
            V = value_iteration(env, gamma);
            vi_policy = extract_policy(optimal_v, gamma)
        else:
            optimal_policy = policy_iteration(env, gamma)
            V = compute_policy_v(env,optimal_policy,gamma)
        df = df.append(pd.DataFrame({'gamma':[gamma for i in range(0,env.observation_space.n)],'state':[i for i in range(0,env.observation_space.n)],'value': V}))
    df.state = df.state.astype(int)
    return df



if __name__ == '__main__':
    env_name  = 'FrozenLake-v0'
    gamma = 0.9
    learning_rate = 0.85
    episodes = 3000

    env = gym.make(env_name,desc=random_map)
    my_seed = 0
    np.random.seed(my_seed)
    random.seed(my_seed)
    env.seed(my_seed)
    env.action_space.seed(my_seed)
    env.observation_space.seed(my_seed)
    ##env.reset()
    optimal_policy = policy_iteration(env, gamma)
    pi_policy = compute_policy_v(env,optimal_policy,gamma)
    print "Policy from Policy Iteration"
    print pi_policy
    scores = evaluate_policy(env, optimal_policy, gamma,n=100)
    print('Average scores,PI = ', scores)
    #display_iteration(optimal_policy,pi_policy,env,"Policy Iteration")


    optimal_v = value_iteration(env, gamma);
    print "Optimal V"
    print optimal_v
    vi_policy = extract_policy(optimal_v, gamma)
    print ("Policy from vi")
    print vi_policy
    vi_policy_score = evaluate_policy(env, vi_policy, gamma, n=100)
    print('Policy average score,VI = ', vi_policy_score)
    #display_iteration(vi_policy,optimal_v,env,"Value Iteration")
    
    #Is the Policy generated by Value and Policy Iteration are the same
    are_same = np.array_equal(vi_policy,optimal_policy)
    print "@@@@@@@@@@@@@Are the Policies Same@@@@@@@@"
    print are_same

    qtable,qpolicy = qlearning(learning_rate,gamma,episodes,env)
    #print "Q Table"
    #print qtable
    print "Q Policy"
    print qpolicy
    qpolicy_score = evaluate_policy(env,qpolicy,gamma,n=100)
    print('Policy average score, QLearning = ',qpolicy_score)
    #display_qiteration(qpolicy,env,"QLearning")


    ##Hyper parameter Learning
    data_frame = gammas_training(env,"value-iteration")
    #print "Gamma Data Frame"
    #print data_frame
    fig,ax = plt.subplots(2,2,figsize=(20,10))
    sns.lineplot(data=data_frame,x='gamma',y='value',hue='state',ax=ax[0][0])
    ax[0][0].set_title('4x4 - Value Iteration - Values per gamma')

    df_policy = gammas_training(env,"policy-iteration")
    sns.lineplot(data=df_policy,x='gamma',y='value',hue='state',ax=ax[0][1])
    ax[0][1].set_title('4x4 - Policy Iteration - Values per gamma')
    plt.savefig("Gamma")
    plt.show()
