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

        self.gamma = 0.5
        self.alpha = 0.2
        self.reach = []
        
        self.SA = {} #a table of frequencies for state-action pairs
        self.NE = 3  # try each action-state pair at least NE time
        self.reward_plus = 2
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def inputs_filter(self, inputs):
        # Filter Rules:
        # inputs = {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}

        inputs['right'] = None         # remove right traffic

        light = inputs['light']

        if light == 'green':          # if light is green, remove left traffic
            inputs['left'] == None
        
        if light == 'red':           # if light is red, replace left traffic status to None if it's going right
            left_waypoint = inputs['left']
            if left_waypoint == 'right':
                inputs['left'] = None
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
        
        #print q_state_action_values
        #print self.SA
        
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
    print a.q_table
    print "Size of q_table = {}, fail rate = {}%".format(len(a.q_table), 100 * a.reach.count(0) / float(len(a.reach)))

if __name__ == '__main__':
    run()
#{'red#left#right#*#forward': [0.21399582165652042, 0, 0, 0], 'red#*#left#*#right': [1.3438397017319341, 0.8510042343712707, 0.8877784637002248, 4.148168757194126], 'red#right#*#*#right': [1.3013446593426365, 0.7387112871032703, 0.8236741007798505, 4.635824116130701], 'green#*#*#left#forward': [0.10816192036391448, 5.156820974522299, 0, 0], 'green#*#left#*#left': [0.12589212699007685, 0.6260445918321444, 8.235080884828754, 0], 'red#*#right#*#right': [1.1285900118998786, 0.6475193787722021, 0.7740987032093045, 3.8778985070671976], 'red#left#*#right#forward': [0.0, 0, 0, 0], 'green#*#*#*#forward': [0.0, 6.467462471732415, 0, 0], 'red#*#forward#left#forward': [0.12515336475153221, 0, 0, 0], 'red#*#forward#*#forward': [1.0497089767495305, -0.003335596550108474, 0.32691548754982314, 0.5130684334083109], 'green#left#*#right#forward': [0.12237855257968751, 0, 0, 0], 'green#forward#*#*#forward': [1.5577081529441745, 4.144394842006355, 0, 0], 'green#*#right#left#right': [0.5895911469257132, 0, 0, 0], 'red#*#*#left#right': [0.6303961966174804, 0.6942115980014002, 0.7666405767761261, 3.952429051303703], 'green#forward#*#left#forward': [0.519463296459884, 0, 0, 0], 'green#right#*#*#left': [1.701106591937549, 0.9469056460101841, 0, 0], 'red#left#*#*#left': [0.4331296317744906, -0.3788585689872461, -0.456304804360483, 1.299993938297257], 'red#forward#*#*#forward': [0.5587790324473995, -0.2921047585912877, -0.1518009222179517, 1.4750566798602551], 'red#*#right#*#left': [0.887672123724325, -0.02997396157598211, 0.21298760278363635, 0], 'red#forward#*#right#forward': [0.09584415424058786, 0, 0, 0], 'red#*#*#*#left': [0.0, -0.692180163624, -0.3911499794281595, 2.042660093585707], 'red#*#right#*#forward': [0.39002522484109386, 0.7108837326360194, -0.20246368647960394, 1.0786231879963228], 'red#*#*#*#forward': [2.0150303065266995, -0.75024, -0.5500800000000001, -0.30724185569920004], 'green#right#*#*#forward': [1.2486576080338194, 5.919226287426235, 0, 0], 'green#left#left#*#forward': [0.13994707771541082, 0, 0, 0], 'green#*#*#forward#left': [0.2547536603570865, 1.0098340705369449, 3.735330268013618, 0], 'red#forward#*#left#forward': [0.6654494037167246, 0, 0, 0], 'green#*#forward#forward#forward': [0.10844817560718206, 1.8374935128251892, 0, 0], 'red#*#*#right#left': [0.6077240826240835, -0.08011004479609132, 0, 0], 'green#*#left#right#forward': [0.504793379709136, 0, 0, 0], 'green#*#right#*#forward': [0.1864369913881157, 4.216782688908991, 0, 0], 'green#*#*#right#forward': [1.603371479900253, 6.68529905311593, 0, 0], 'green#*#forward#*#forward': [0.20016458203279136, 4.53102758140686, 0, 0], 'green#left#forward#*#forward': [0.17341897515902055, 0.553897981475894, 0, 0], 'red#*#*#forward#right': [1.2258191372257052, 0.589835039104639, 0.7409608097381617, 2.8515504064814223], 'red#right#*#*#left': [0.4647840940763574, 0.3011763891813194, 0.2697049132505059, 0.3726974005413797], 'red#*#*#left#forward': [0.6980446434044549, 0.1601907694335641, 0.2913133628767334, 1.487530123504517], 'red#forward#*#*#left': [0.48620042443791445, -0.39305946182028867, 0.13736588618086382, 1.9325663854034965], 'green#right#*#left#forward': [0.41687392273981416, 0, 0, 0], 'red#left#*#*#right': [0.0, -0.7037886761014466, -0.6650410919541013, 4.530997376808697], 'red#*#*#forward#left': [0.49748478913382976, -0.034129581439607, 0, 0], 'red#forward#*#right#right': [0.3742552721035251, 0, 0, 0], 'red#*#forward#*#right': [1.4345948873041299, 0.6078237101290629, 0.7517697341750877, 5.17909662414395], 'red#right#*#*#forward': [2.1797336762725403, 0.20796242145867555, 0.06915012282192913, 0.5704418791673047], 'red#*#right#left#right': [0.3332341188030458, 0, 0, 0], 'green#right#forward#*#right': [0.43264414481444624, 0, 0, 0], 'green#*#left#left#right': [0.5335767867553719, 0.0680215137252117, 0, 0], 'red#*#*#left#left': [0.6720811298978113, 0.04864465845453332, -0.0106595849743105, 0], 'green#left#*#*#forward': [0.0, 4.955957285128386, 0, 0], 'green#left#*#forward#left': [0.4133134711722442, 0, 0, 0], 'green#left#*#left#forward': [0.155192989007449, 0, 0, 0], 'green#*#forward#*#left': [0.24935197091599942, 0.5238865736157013, 5.637996050847327, 1.3424593059576841], 'green#left#right#*#forward': [0.1322918451057284, 3.193235567095489, 0, 0], 'green#*#left#forward#forward': [0.20907093912261387, 1.0221349254506051, 0, 0], 'green#forward#left#*#right': [0.38463717161514366, 0, 0, 0], 'green#left#*#*#right': [0.0, 1.0358727919523616, 1.1084068134902627, 3.9464464801869443], 'green#*#*#left#right': [0.6369977354765868, 0.9791854220903713, 0.8159536853739766, 5.205070721498437], 'green#*#left#*#forward': [0.08200500233716897, 5.095014959502427, 0, 0], 'green#*#*#*#left': [0.0, -0.37512, 5.592063572513188, 0.2883501921024001], 'red#*#*#right#right': [1.391099285484215, 0.6772473320403808, 0.8534947315609396, 3.930220569736433], 'red#*#left#*#left': [0.7366921936968245, -0.01625417425612541, 0.5485020633672715, 0], 'red#*#*#forward#forward': [0.3425337884566611, 0.13751260743393975, 0.6341305311779386, 2.0467520020594], 'green#*#forward#left#forward': [0.21403906540985207, 0, 0, 0], 'red#*#right#right#right': [0.40712393445486317, 0, 0, 0], 'red#forward#*#left#right': [0.4393591789560148, 0, 0, 0], 'red#*#*#*#right': [0.0, -0.7437600000000002, -0.7538284800000001, 4.064890375587707], 'red#forward#*#*#right': [0.2899152833454677, -0.18841451630729408, -0.4656892579962144, 4.569399896548266], 'red#left#*#forward#forward': [0.2036627633803538, 0, 0, 0], 'green#right#*#*#right': [1.5803894254867692, 1.2647818875177437, 0.7381604576326273, 3.8848351913170935], 'green#forward#*#*#left': [1.5008308907230599, 1.0484784867742991, 0.7969750027123902, 0], 'green#*#left#left#forward': [0.09379683932049826, 5.390296857248124, 0, 0], 'green#*#left#*#right': [0.2584391977777628, 1.1013673971739522, 0.8755395384391818, 4.577723558046665], 'green#*#*#left#left': [0.14358203805773198, 0.9977817171082497, 4.784849054734169, 0], 'green#left#forward#*#right': [0.3599504380982186, 0, 0, 0], 'red#*#*#right#forward': [1.5018813903405015, 0.3695768881724203, 0.7882582645652048, 0.4323032723205131], 'green#*#*#right#right': [1.4924708315235267, 1.1946169976232885, 1.2233278897887836, 2.7299009241519885], 'green#*#*#forward#right': [0.6564263141314439, 1.2326570031194755, 0.8776851570777814, 4.675091961650923], 'green#*#forward#*#right': [0.68642989513162, 0.9090110305141652, 0.8318002306251173, 4.139097635957697], 'red#*#forward#*#left': [0.4502084698534972, -0.030790804195038615, 0.2365325716250772, 0], 'green#*#forward#forward#right': [0.7198538850279228, 0, 0, 0], 'green#*#right#*#left': [1.4203971186003774, 1.0652662309069207, 3.900909044345768, 0], 'green#forward#*#*#right': [1.3755214527510171, 0.980594176682666, 0.6339500682695542, 4.044251290247754], 'green#*#left#left#left': [0.31017917740056017, 0, 0, 0], 'green#*#right#*#right': [1.3949377287904037, 1.055965912428512, 1.027115214861458, 4.4441712283730475], 'green#forward#forward#*#forward': [0.868859282777094, 0, 0, 0], 'green#*#*#*#right': [0.0, -0.21300480000000005, 0.20404324722560013, 4.230485088721513], 'green#*#*#forward#forward': [0.12823200935424864, 5.025005156210336, 0, 0], 'red#forward#left#*#forward': [0.1382705100959631, 0, 0, 0], 'red#*#forward#right#forward': [0.14945340303797264, 0, 0, 0], 'green#left#*#*#left': [1.9529360202933677, 1.1519368341083003, 4.395988780191176, 0], 'red#*#left#forward#right': [0.35260353232694963, 0, 0, 0], 'green#*#*#right#left': [1.8476831119320551, 1.0159498137920475, 5.644261854303592, 0], 'red#left#forward#*#forward': [0.16377540350721453, 0, 0, 0], 'green#forward#right#*#forward': [0.6981596299009434, 0, 0, 0], 'red#left#*#*#forward': [0.0, -0.6386160713825909, 0.1468485413440631, 1.3664661855880775], 'red#right#forward#*#right': [0.5036166307367694, 0, 0, 0], 'red#*#left#*#forward': [0.5699590326122579, 0.2088678628075769, -0.3343256370952581, 0.7945265620910511]}
#Size of q_table = 97, fail rate = 0.83%
