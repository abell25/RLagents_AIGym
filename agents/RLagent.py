from abc import ABCMeta, abstractmethod, abstractproperty

class RLagent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def update(self, prev_obs, action, reward, obs):
        pass

    # TODO: this continuous -> discrete transformation should be done by some kind of state quantizer instead
    #      so the agent only sees the discrete states. (but will need to notify when a new state emerges?)
    def s_to_discrete(self, obs):
        """ converts continuous stats (obs) into a discrete state representation.

            ( some agents do not need a discrete state representation though)
        """