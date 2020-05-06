
import numpy as np
import tensorflow as tf

#Game Param
SCREENWIDTH = 1250
SCREENHEIGHT = 1250



should_load_ = False
model_itr = '_final'

width = 600 #1000#600    - 1000
w = 50 #12#60     - 20
coords_width = width/w
agentSpeed = 12 #60

Talk_range = 20

Num_Mammoth = 2 #1
Num_Tiger = 0 #np.random.randint(0, 2)#2
Num_Insects = 3 #2
Num_Monkey = 2 #np.random.randint(2, 5)#2
Num_Fruits = 10 #np.random.randint(20, 40)#13
Num_P_Fruits = 0 #5
Num_Mouse = 1 #2
Num_Deer = 1 #2

Num_Agents = 2#1
human_agent_on = False
range_render_obst = False
range_render = False
#Gloabal Variables
success_prob = 0.98

#Containers
obstacle_type1_1 = (1, 0)
obstacle_type1_2 = (2, 0)

obstacle_type2_1 = (0, 1)
obstacle_type2_2 = (0, 2)
Obst_likelyhood = 1#2 #5

CoordList = []
TypeList_obst = []
CoordList_obst = []
CoordList_obst_detail = []

CreatureList = []
AgentList = []
ExistenceList = []

Creature_Name_to_typeId = {'mammoth': 1, 'tiger': 2, 'insects': 3, 'deer': 4,
                           'mouse': 5, 'monkey': 6, 'fruit': 7, 'p-fruit': 8,
                           'agent1': 9, 'agent2': 10, 'agent3': 11}

showing_mode = [0, True, True, True, True, True, True, True, True, True, True, True]

Human_type = 12

#init_hp = [100, 30, 1, 10, 5, 10, 1, 1]
#max_speed = [3, 4, 2, 4, 2, 2, 0, 0]
#damage = [10, 10, 0.05, 1, 1, 4, 0, 0]

init_hp = [100, 20, 1, 10, 5, 10, 1, 1]
max_speed = [3, 2, 2, 4, 2, 2, 0, 0]
damage = [10, 2, 0.05, 1, 1, 4, 0, 0]
init_cal = [100, 20, 2, 20, 5, 7, 5, -20]

angles_ = [180, 0, 90, -90]


#vocab idx = 0 ~ 23
Vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 'x', 't', ' ', '.', '!', '@', '..'] # X Number of Agent

Emotions = {'A', 'B', 'C', 'D', 'E', 'F', 'G', '-'} # X {low, midd, high} X Number of Agent

abs_coords = np.zeros((int(coords_width), int(coords_width)), dtype=object)


policy_learner = None

#Tensor Params
#Training Param and Variables
Lam0 = 1# 2 ->10
Lam1 = 0.1 #.001#0.001 #0 ~ 0.001 for Policy Entropy
Lam2 = 0.1
BatchSize = 1000
NE = 10

is_learner_set = False

IntentionNoiseDim = 4
IntentionDim1 = 2
Maximum_inventory = 5
Maximum_object_recognition = 45 #200 ->45
StateDim = 2 + (Maximum_inventory + 1) + Maximum_object_recognition
ActionDim = 6 #4-6
State_and_Action_Dim = StateDim + ActionDim
init_Epsil = 98#36

Epsilon = init_Epsil
Whether_to_use_language = True

#########################################################################################

init_prev_A = np.random.randint(0, 8)
init_prev_B = np.random.randint(0, 8)
init_prev_C = np.random.randint(0, 8)
prev_actions = [init_prev_A, init_prev_B, init_prev_C]

observed_state_type_stacked = [[], [], []]
observed_state_actions_took_stacked = [[], [], []]
observed_state_hp_stacked = [[], [], []]
observed_state_angle_stacked = [[], [], []]
said_word_history_stcked_agent1 = [[], [], []]
said_word_history_stcked_agent2 = [[], [], []]
said_word_history_stcked_agent3 = [[], [], []]
emotion_stacked_agent1 = [[], [], []]
emotion_stacked_agent2 = [[], [], []]
emotion_stacked_agent3 = [[], [], []]
reward_stacked = [[], [], []]
reward_stacked_com = [[], [], []]
hf_value_stacked = [[], [], []]



observed_state_type_stacked_big = [[], [], []]
observed_state_actions_took_stacked_big = [[], [], []]
observed_state_hp_stacked_big = [[], [], []]
observed_state_angle_stacked_big = [[], [], []]
said_word_history_stcked_agent1_big = [[], [], []]
said_word_history_stcked_agent2_big = [[], [], []]
said_word_history_stcked_agent3_big = [[], [], []]
emotion_stacked_agent1_big = [[], [], []]
emotion_stacked_agent2_big = [[], [], []]
emotion_stacked_agent3_big = [[], [], []]
reward_stacked_big = [[], [], []]
reward_stacked_com_big = [[], [], []]
hf_value_stacked_big = [[], [], []]
