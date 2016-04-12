import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.status_log = {} #state string, action_index, reward
        self.q_table = {} # state string -> q_value list
        # valid_actions = [None, 'forward', 'left', 'right']
        self.actions = env.valid_actions

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def next_action_index(self, current_state):
        if current_state not in self.q_table.keys():
            q_table[current_state]= [0, 0, 0, 0]
        max_action_index = self.q_table[current_state].index(max(q_table[current_state]))
        
        # TODO: if has several max, return a random one
        return max_action_index

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # input = {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        # TODO: Update state
        current_state = (inputs.values() + [self.next_waypoint]).join('#')
        self.status_log[t] =  tuple(current_state, action, reward)
        
        # choose next action
        # next_action = random.choice(self.actions)
        max_action_index = self.next_action_index(current_state)
        # Execute action and get reward
        reward = self.env.act(self, self.actions[max_action_index])

        # TODO: Learn policy based on state, action, reward
        # Q(state, action) = Q(state, action) + alpha * (R(state, action) + Gamma * Max[Q(next state, all actions)])
        if (t != 0):
            

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
    current_state = 
    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
