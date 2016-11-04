
from __future__ import print_function

#import sys
#sys.path.insert(0, r'/Users/anthbell/RL/gym')
import gym

def debug(str):
    DEBUG=True
    if DEBUG:
        print(str)

from agents.Qagent import Qagent

f = open('./numbers.txt', 'w')
#env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0') # works with Qagent
env = gym.make('Acrobot-v1')
#env = gym.make('Pendulum-v0')
agent = Qagent(actions=sorted(list(set(env.action_space.sample() for _ in range(1000)))))
actions = []
done_count = 0
last_num_iters = -99999
for episode_i in xrange(100000):
    obs = env.reset()
    agent.reset()
    done_iters = 0
    done_i = 0
    for t in xrange(1000000):
        if episode_i % 100 == 0 and episode_i > 0:
            env.render()
        action = agent.get_action(obs)
        actions.append(action)
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        ##debug("ep: {:3}/{:4} done:{:2}, R: {:.2f}, S: {}, A: {} L/R: {:.2f}/{:.2f}, lst:{:5}".format(episode_i, t, done_count, reward, obs, action, agent.qs[0][1], agent.qs[1][1], last_num_iters))

        if t % 500 == 0:
            debug("{:3}/{:4} done:{:2}, R:{:.2f}, S:{}, A:{}, eta:{:.2f}, lst:{:5}, Q:{}, |S|={}".format(episode_i, t, done_count, reward, obs, action, agent.eta, last_num_iters, " ".join(["{}:{:.2f}".format(n[0],n[1]) for n in agent.qs]), len(agent.points)))

        if done:
            reward = 0.0
        agent.update(prev_obs, action, reward, obs)

        if done and done_i == 0:
            done_count += 1
            last_num_iters = t
            s_discrete = agent.s_to_discrete(obs)
            if episode_i % 1 == 0:
                print("\n")
                print("Episode {} finished after {} steps, score: {:.2f}, s: {}, len(Q): {}".format(episode_i, t+1, agent.score, s_discrete, len(agent.Q)))
                print("R:{:.2f}, S:{}, A:{}, eta:{:.2f}, Q:{}".format(reward, obs, action, agent.eta, " ".join(["{}:{:.2f}".format(n[0],n[1]) for n in agent.qs])))
                print("\n")

            f.write(str(t) + '\n')

        if done:
            debug("DONE, iter: {}, reward: {}".format(done_i, reward))
            done_i += 1
            if done_i >= done_iters:
                done_i = 0
                break
