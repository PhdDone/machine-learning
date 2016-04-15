import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.status_log = {} #state string, action_index, reward
        self.q_table = {} # state string -> q_value list
        self.actions = env.valid_actions

        self.gamma = 0.7
        self.alpha = 0.5
        self.reach = []
        
        self.SA = {} #a table of frequencies for state-action pairs
        self.NE = 5  # try each action-state pair at least NE time
        self.reward_plus = 2
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def next_action_index(self, current_state):
        if current_state not in self.q_table.keys():
            self.q_table[current_state]= [0, 0, 0, 0]
        
        max_action_index = self.q_table[current_state].index(max(self.q_table[current_state]))
        
        action_index_candidates = [max_action_index]
        for i in xrange(len(self.q_table[current_state])): #randomly choose next action if there're several max
            if i != max_action_index:
                if self.q_table[current_state][i] == self.q_table[current_state][max_action_index]:
                    action_index_candidates.append(i)
        if len(action_index_candidates) > 1:
            max_action_index = random.choice(action_index_candidates)
        return max_action_index

    def next_action_index_with_exploration(self, current_state):
        if current_state not in self.q_table.keys():
            self.q_table[current_state]= [0, 0, 0, 0]
        max_action_index = self.q_table[current_state].index(max(self.q_table[current_state]))

        q_state_action_values = []
        
        for i in xrange(len(self.actions)):
            SA_key = current_state + str(i)
            
            if SA_key not in self.SA.keys():
                self.SA[SA_key] = 0

            if self.SA[SA_key] >= self.NE: # already tried NE times, use the value from q_table
                q_state_action_values.append(self.q_table[current_state][i])
            else:
                q_state_action_values.append(self.reward_plus) #haven't tried enough? give a high reward
            
        
        max_action_index = q_state_action_values.index(max(q_state_action_values))
        
        #print q_state_action_values
        #print self.SA
        
        SA_key = current_state + str(max_action_index)
        self.SA[SA_key] += 1 #update SA map

        return max_action_index

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # inputs = {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        # TODO: Update state
        #current_state = ('#').join((inputs.values() + [self.next_waypoint]))
        current_state = "#".join(map(lambda x : x if x is not None else "*", inputs.values()) + [self.next_waypoint])
        # choose next action
        # next_action = random.choice(self.actions)
        max_action_index = self.next_action_index_with_exploration(current_state)
        #max_action_index = self.next_action_index(current_state)
        #print current_state + str(max_action_index)
        # Execute action and get reward
        reward = self.env.act(self, self.actions[max_action_index])
        self.status_log[t] =  [current_state, max_action_index, reward]
        # TODO: Learn policy based on state, action, reward
        # Q(state, action) = (1 - alpha) * Q(state, action) + alpha * (R(state, action) + Gamma * Max[Q(next state, all actions)])
        # state : the prev state
        # action: the prev action

        if t != 0:
            prev_state = self.status_log[t-1][0]
            prev_action_index = self.status_log[t-1][1]
            prev_reward = self.status_log[t-1][2]
            self.q_table[prev_state][prev_action_index] = (1-self.alpha)*self.q_table[prev_state][prev_action_index] \
                + self.alpha * (prev_reward + self.gamma * self.q_table[current_state][max_action_index])

        if reward > 2 or deadline == 0:
            self.q_table[current_state][max_action_index] = (1 - self.alpha) * self.q_table[current_state][max_action_index] + self.alpha * reward
            self.reach.append(deadline)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_state(self):
        return 0

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print len(a.q_table)
    print np.mean(a.reach)
    print a.reach.count(0) / float(len(a.reach))

if __name__ == '__main__':
    run()
#{'green#*#*#right#forward': [0, 0, 2.7365205977812534, 0], 'green#*#forward#*#forward': [0, 0, 0, 3.362099303625455], 'green#left#forward#*#forward': [0.0, 0, 0, 1.5101926930964797], 'green#*#*#left#forward': [0.0, 6.779064491505034, 0, 0], 'green#*#left#*#left': [0, 0, 4.8258255089266955, 0], 'green#*#left#*#forward': [0, 8.559187679171492, 0, 0], 'green#*#*#*#left': [0.0, -0.25, 7.3434242928969535, 0], 'red#*#left#*#forward': [0, 0, 1.453232153731208, 0], 'red#*#*#forward#right': [0, 0, 0, 2.4137009547336277], 'green#*#*#*#forward': [0, 9.31187743718652, -0.25, 0], 'red#*#forward#*#forward': [0, 0, 0, 2.047117653136957], 'red#*#*#forward#forward': [0, 0, 0, 1.5244705525324962], 'red#*#*#left#forward': [0, 0, 0, 3.792610411863949], 'green#forward#*#*#forward': [0, 0, 0, 0.9941256173075301], 'red#left#*#*#right': [0, 0, 0, 4.309458546959114], 'red#left#*#*#forward': [3.601695860702714, 0, 0, 0], 'red#right#*#*#forward': [1.7347456661452227, 0, 0, 0], 'red#*#*#*#right': [0, 0, 0, 5.190005882814992], 'red#*#right#*#left': [0, 0, 0, 1.8774019094672547], 'red#forward#*#*#forward': [0, 0, -0.5, 3.8968658915000844], 'green#*#*#*#right': [0, -0.25, -0.25, 8.10881673548693], 'green#*#*#forward#forward': [0, 6.637784105058999, 0, 0], 'green#left#*#*#forward': [0.0, 7.078262249123607, 0, 0], 'red#*#*#*#left': [3.731354342461704, -0.5, -0.5, 0], 'red#*#right#*#forward': [0, 0, 0, 1.311811211506034], 'green#left#*#*#left': [0, 2.662276343818317, 0, 0], 'red#*#forward#*#left': [0, 0, 1.1002791236840526, 0], 'green#right#*#*#forward': [0, 0, 0, 2.9416670118642934], 'red#*#*#*#forward': [3.162557030118079, -0.5, -0.5, 0], 'green#*#*#forward#left': [0, 0, 0, 2.571679920145685], 'green#*#forward#*#left': [0.0, 0, 6.463339206356009, 0], 'green#*#*#left#left': [0, 0, 0, 1.9865799684231171], 'red#*#*#right#forward': [0, 1.1293313108248246, 0, 0], 'green#*#right#*#forward': [0, 1.5539819943089355, 0, 0]}
#17.44
