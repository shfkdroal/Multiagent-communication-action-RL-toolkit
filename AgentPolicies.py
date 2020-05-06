import numpy as np
from Game_Param import *
import random
import copy
from Agent_Net import *

# the function modifies the word to say, and emotion to express
def agent_policy1(type_input, hp_input, angle_input, action_took_input,
                                       hf_value, words_input1, words_input2, words_input3,
                                       emotion_input1, emotion_input2,
                                       emotion_input3, learner, self, is_pred_mode):


    com_action, action, internal_mentality = learner.test_one_patch(type_input=type_input, hp_input=hp_input, angle_input=angle_input,
                           action_took_input=action_took_input, hf_value=hf_value, words_input1=words_input1,
                           words_input2=words_input2, words_input3=words_input3, emotion_input1=emotion_input1,
                           emotion_input2=emotion_input2, emotion_input3=emotion_input3, is_training=True)


    if is_pred_mode:
        com_action_values, action_values = copy.deepcopy(com_action), copy.deepcopy(action)
        com_action_values, action_values = np.max(com_action_values), np.max(action_values)
        return action_values, com_action_values

    #execution mode below
    # real_com_action_ = copy.deepcopy(real_com_action)
    # real_com_action_ = np.argmax(real_com_action_)
    # what_to_say_or_express = real_com_action_ // Num_Agents

    coin = np.random.randint(0, 100)
    self.said_word = np.random.randint(0, len(Vocab) - 1)  # np.random.randint(0, 3)
    self.expressed_emotion = np.random.randint(0, len(Emotions) - 1)

    if Epsilon >= coin:
        coin_com_action_whether_to_express_or_talk = random.randint(0, 2)
        if coin_com_action_whether_to_express_or_talk == 1:
            com_action = np.random.randint(10, 10 + Num_Agents)
            self.speech_target = -1
            self.expression_target = com_action - 10
            self.expressed_emotion = np.random.randint(0, len(Emotions))
            #print("rand express Agent1")
        else:
            #print("rand chat Agent1")
            com_action = np.random.randint(7, 7 + Num_Agents)
            self.speech_target = com_action - 7
            self.expression_target = -1
            self.said_word = np.random.randint(0, len(Vocab))#np.random.randint(0, 3)
        action = np.random.randint(0, 8)
        if prev_actions[0] < 4:
            action = np.random.randint(4, 8)
        prev_actions[0] = action

        #self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        intensity = np.random.randint(0, 3)
        self.emotion_intensity = intensity

    else:
        action = np.argmax(action) // Num_Agents
        # print(com_action)
        com_action = np.argmax(com_action)

        what_to_say_or_express = com_action // Num_Agents
        agent_target = com_action % Num_Agents

        intensity = 0
        word_voc = len(Vocab) - 1
        emotion_voc = len(Emotions) - 1
        if what_to_say_or_express >= len(Vocab):
            intensity = (com_action // len(Emotions)) + 1
            emotion_voc = com_action % len(Emotions)
            com_action = agent_target + 10  # emotional communication
            self.speech_target = -1
            self.expression_target = agent_target
        else:
            word_voc = what_to_say_or_express
            com_action = agent_target + 7  # verbal communication
            self.speech_target = agent_target
            self.expression_target = -1

        #self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        self.emotion_intensity = intensity
        self.expressed_emotion = emotion_voc
        self.said_word = word_voc

    global Whether_to_use_language
    if not Whether_to_use_language:
        com_action = 0
        self.emotion_intensity = 0
        self.expressed_emotion = len(Emotions) - 1
        self.said_word = len(Vocab)-1

    return action, com_action



def agent_policy2(type_input, hp_input, angle_input, action_took_input,
                                       hf_value, words_input1, words_input2, words_input3,
                                       emotion_input1, emotion_input2,
                                       emotion_input3, learner, self, is_pred_mode):

    com_action, action, internal_mentality = learner.test_one_patch2(type_input=type_input, hp_input=hp_input,
                                                                    angle_input=angle_input,
                                                                    action_took_input=action_took_input,
                                                                    hf_value=hf_value, words_input1=words_input1,
                                                                    words_input2=words_input2,
                                                                    words_input3=words_input3,
                                                                    emotion_input1=emotion_input1,
                                                                    emotion_input2=emotion_input2,
                                                                    emotion_input3=emotion_input3, is_training=True)
    if is_pred_mode:
        com_action_values, action_values = copy.deepcopy(com_action), copy.deepcopy(action)
        com_action_values, action_values = np.max(com_action_values), np.max(action_values)
        return action_values, com_action_values

    # execution mode below

    coin = np.random.randint(0, 100)
    self.said_word = np.random.randint(0, len(Vocab) - 1)  # np.random.randint(0, 3)
    self.expressed_emotion = np.random.randint(0, len(Emotions) - 1)

    if Epsilon >= coin:
        coin_com_action_whether_to_express_or_talk = random.randint(0, 2)
        if coin_com_action_whether_to_express_or_talk == 1:
            com_action = np.random.randint(10, 10 + Num_Agents)
            self.speech_target = -1
            self.expression_target = com_action - 10
            self.expressed_emotion = np.random.randint(0, len(Emotions))
            # print("rand express Agent1")
        else:
            # print("rand chat Agent1")
            com_action = np.random.randint(7, 7 + Num_Agents)
            self.speech_target = com_action - 7
            self.expression_target = -1
            self.said_word = np.random.randint(0, len(Vocab))  # np.random.randint(0, 3)
        action = np.random.randint(0, 8)
        if prev_actions[1] < 4:
            action = np.random.randint(4, 8)
        prev_actions[1] = action

        # self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        intensity = np.random.randint(0, 3)
        self.emotion_intensity = intensity

    else:
        action = np.argmax(action) // Num_Agents
        # print(com_action)
        com_action = np.argmax(com_action)

        what_to_say_or_express = com_action // Num_Agents
        agent_target = com_action % Num_Agents

        intensity = 0
        word_voc = len(Vocab) - 1
        emotion_voc = len(Emotions) - 1
        if what_to_say_or_express >= len(Vocab):
            intensity = (com_action // len(Emotions)) + 1
            emotion_voc = com_action % len(Emotions)
            com_action = agent_target + 10  # emotional communication
            self.speech_target = -1
            self.expression_target = agent_target
        else:
            word_voc = what_to_say_or_express
            com_action = agent_target + 7  # verbal communication
            self.speech_target = agent_target
            self.expression_target = -1

        # self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        self.emotion_intensity = intensity
        self.expressed_emotion = emotion_voc
        self.said_word = word_voc

    global Whether_to_use_language
    if not Whether_to_use_language:
        com_action = 0
        self.emotion_intensity = 0
        self.expressed_emotion = len(Emotions) - 1
        self.said_word = len(Vocab) - 1

    return action, com_action


def agent_policy3(type_input, hp_input, angle_input, action_took_input,
                                       hf_value, words_input1, words_input2, words_input3,
                                       emotion_input1, emotion_input2,
                                       emotion_input3, learner, self, is_pred_mode):

    com_action, action, internal_mentality = learner.test_one_patch3(type_input=type_input, hp_input=hp_input,
                                                                    angle_input=angle_input,
                                                                    action_took_input=action_took_input,
                                                                    hf_value=hf_value, words_input1=words_input1,
                                                                    words_input2=words_input2,
                                                                    words_input3=words_input3,
                                                                    emotion_input1=emotion_input1,
                                                                    emotion_input2=emotion_input2,
                                                                    emotion_input3=emotion_input3, is_training=True)
    if is_pred_mode:
        com_action_values, action_values = copy.deepcopy(com_action), copy.deepcopy(action)
        com_action_values, action_values = np.max(com_action_values), np.max(action_values)
        return action_values, com_action_values

    # execution mode below

    coin = np.random.randint(0, 100)
    self.said_word = np.random.randint(0, len(Vocab) - 1)  # np.random.randint(0, 3)
    self.expressed_emotion = np.random.randint(0, len(Emotions) - 1)

    if Epsilon >= coin:
        coin_com_action_whether_to_express_or_talk = random.randint(0, 2)
        if coin_com_action_whether_to_express_or_talk == 1:
            com_action = np.random.randint(10, 10 + Num_Agents)
            self.speech_target = -1
            self.expression_target = com_action - 10
            self.expressed_emotion = np.random.randint(0, len(Emotions))
            # print("rand express Agent1")
        else:
            # print("rand chat Agent1")
            com_action = np.random.randint(7, 7 + Num_Agents)
            self.speech_target = com_action - 7
            self.expression_target = -1
            self.said_word = np.random.randint(0, len(Vocab))  # np.random.randint(0, 3)
        action = np.random.randint(0, 8)
        if prev_actions[2] < 4:
            action = np.random.randint(4, 8)
        prev_actions[2] = action

        # self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        intensity = np.random.randint(0, 3)
        self.emotion_intensity = intensity

    else:
        action = np.argmax(action) // Num_Agents
        # print(com_action)
        com_action = np.argmax(com_action)

        what_to_say_or_express = com_action // Num_Agents
        agent_target = com_action % Num_Agents

        intensity = 0
        word_voc = len(Vocab) - 1
        emotion_voc = len(Emotions) - 1
        if what_to_say_or_express >= len(Vocab):
            intensity = (com_action // len(Emotions)) + 1
            emotion_voc = com_action % len(Emotions)
            com_action = agent_target + 10  # emotional communication
            self.speech_target = -1
            self.expression_target = agent_target
        else:
            word_voc = what_to_say_or_express
            com_action = agent_target + 7  # verbal communication
            self.speech_target = agent_target
            self.expression_target = -1

        # self.internal_mental_state += internal_mentality
        self.internal_mental_state = internal_mentality
        self.emotion_intensity = intensity
        self.expressed_emotion = emotion_voc
        self.said_word = word_voc

    global Whether_to_use_language
    if not Whether_to_use_language:
        com_action = 0
        self.emotion_intensity = 0
        self.expressed_emotion = len(Emotions) - 1
        self.said_word = len(Vocab) - 1

    return action, com_action