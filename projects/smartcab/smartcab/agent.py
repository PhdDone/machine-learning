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
        self.actions = env.valid_actions # valid_actions = [None, 'forward', 'left', 'right']

        self.gamma = 0.1
        self.alpha = 0.2
        self.reach = []
        
        self.SA = {} #a table of frequencies for state-action pairs
        self.NE = 3  # try each action-state pair at least NE time
        self.reward_plus = 2
        self.current_state = ''
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def inputs_filter(self, inputs):
        # Filter Rules:
        # inputs = {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        inputs['right'] = None         # ignore right traffic
        light = inputs['light']

        if light == 'green':          # if light is green, ignore left traffic
            inputs['left'] = None
        
        if light == 'red':
            #left_waypoint = inputs['left']
            #if left_waypoint == 'right':
            #    inputs['left'] = None
            inputs['oncoming'] = None # if light is red, ignore oncoming traffic (assume others follow rules)
        return inputs

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
        
        
        SA_key = current_state + str(max_action_index)
        self.SA[SA_key] += 1 #update SA map

        return max_action_index

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        inputs = self.inputs_filter(inputs)
        
        deadline = self.env.get_deadline(self)
        
        # inputs = {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        # TODO: Update state
        #current_state = ('#').join((inputs.values() + [self.next_waypoint]))
        current_state = ''
        for key in inputs.keys():
            status = inputs[key]
            if status == None:
                status = "*"
            current_state = current_state + '#' + key + ":" + status
        current_state = current_state + "#waypoint:" + self.next_waypoint
        self.current_state = current_state
        #current_state = "#".join(map(lambda x : x if x is not None else "*", inputs.values()) + [self.next_waypoint])
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
        
        # Terminal
        # Since already called self.env.act(), program will terminate
        # But still need to update q_table, current_state becomes prev_state, max_action_index becomes
        # prev_action_index, reward becomes prev_reward
        # Max[Q(next state, all actions)] = Q(terminal, None) = reward.

        if reward > 2 or deadline == 0:
            self.q_table[current_state][max_action_index] = (1 - self.alpha) * self.q_table[current_state][max_action_index] + self.alpha * (reward + self.gamma * reward)
            self.reach.append(deadline)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_state(self):
        return self.current_state

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    # print use the last one
    #for key in a.q_table.keys():
    #    print key + " ".join(str(a.q_table[key]))
    print "Size of q_table = {}, fail rate = {}%".format(len(a.q_table), 100 * a.reach.count(0) / float(len(a.reach)))
    #for key in a.q_table.keys():
    #    print key + " ".join(str(a.q_table[key]))
    #return len(a.q_table), 100 * a.reach.count(0) / float(len(a.reach))
def run_multi():
    ave_fr = 0
    ave_q_size = 0
    for x in xrange(100):
        q_size, fr = run()
        ave_fr += fr
        ave_q_size += q_size
    print "ave q size = {}, ave fr = {}%".format(ave_q_size/100, ave_fr/100)
    
if __name__ == '__main__':
    run()
    #run_multi()
