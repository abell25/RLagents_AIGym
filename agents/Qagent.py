
import random
import numpy as np

def debug(str):
    DEBUG=False
    if DEBUG:
        print(str)

class Qagent:
    def __init__(self, epsilon=0.99, init_eta=1.0, min_eta=0.15, stepsize=0.01, grid_radius=0.4, init_q=1.0, actions=None):
    #def __init__(self, epsilon=0.99, eta=0.15, stepsize=0.005, grid_radius=0.001, init_q=1.0, actions=None): # Mountain-car
        """ Qagent

            epsilon: time penalty
            eta: prob of doing exploration step
            stepsize: step-size for updating Q
            grid_radius: size of grid elements (generated dynamically)
            init_q: initial value for Q state-action values
            actions: List of actions available

        """

        self.epsilon = epsilon
        self.eta = init_eta
        self.min_eta = min_eta
        self.stepsize = stepsize
        self.init_q = init_q

        self.actions = actions
        self.Q = {}

        self.s = None
        self.a = None
        self.sars = []
        self.episode_sars = []
        self.episode_qs = []
        self.score = 0.0

        self.points = set()
        self.point_lookup = {}
        self.grid_radius = grid_radius

    def reset(self):
        self.episode_sars = []
        self.s = self.a = None
        self.score = 0.0

    def get_action(self, s):
        self.qs = [(ai, self.calc_Q(s, ai)) for ai in self.actions]

        self.eta = max(self.min_eta, self.eta * 0.99999)

        if random.random() < self.eta:
            a = random.sample(self.actions, 1)[0]
        else:
            a = max(self.qs, key=lambda x: x[1])[0]

        self.s = s
        return a

    def update(self, s, a, r, new_s):
        self.episode_sars.append((s, a, r, new_s))
        self.update_Q(s, a, r, new_s)
        self.score += r

    def update_Q(self, s, a, r, new_s):
        # Q(s,a) <- ( r + epsilon * max_a Q(s',a') ) - Q(s,a)

        s_discrete = self.s_to_discrete(s)
        next_q = max([self.calc_Q(new_s, ai) for ai in self.actions])
        q = self.calc_Q(s, a)

        self.Q[(s_discrete, a)] = q + self.stepsize * ((r + self.epsilon * next_q) - q)
        if self.Q[(s_discrete, a)] is None:
            raise Exception()
        debug("a: {}, r:{:.2f}, q:{:.2f}, next_q:{:.2f}, q_error: {:.2f}, updated_q_est:{:.2f}".format(
            a, r, q, next_q, (( r + self.epsilon * next_q) - q), self.Q[(s_discrete, a)]))

    def calc_Q(self, s, a):
        s_discrete = self.s_to_discrete(s)
        p = (s_discrete, a)
        ## debug("calc_Q:  p in Q?: {}, Q[p]:{:.2f}, p:{}".format(p in self.Q, self.Q[p] if p in self.Q else -99.0, p))
        if p not in self.Q:
            self.Q[p] = self.init_q
            if self.Q[p] is None:
                raise Exception("part b")
        return self.Q[p]

    # TODO: remove this code and have driver use state quantizer instead (or take stateQuantizer as an init parameter)
    def s_to_discrete(self, s):
        # return tuple([np.sign(n) for n in s])
        return tuple(self.get_s_point(s))

    def get_s_point(self, s):
        s_t = tuple(s)
        if s_t in self.point_lookup:
            return self.point_lookup[s_t]

        if s_t in self.points:
            return s

        if len(self.points) == 0:
            self.points.add(s_t)
            self.point_lookup[s_t] = s_t
            return s

        d = lambda x, y: np.sum((x - y) ** 2)
        closest_p = min(self.points, key=lambda x: d(x, s))
        if d(s, closest_p) >= self.grid_radius:
            self.points.add(s_t)
            self.point_lookup[s_t] = s_t
            return s
        else:
            self.point_lookup[s_t] = closest_p
            return closest_p
