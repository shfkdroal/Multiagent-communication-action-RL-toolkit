import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import graphic
from World_manager import *
from Creature_policies import *
from AgentPolicies import *
import copy
import csv
from operator import itemgetter

##########################

wn = turtle.Screen()
wn.setup(SCREENWIDTH, SCREENHEIGHT)
wn.bgcolor("black")
wn.title("Hunt-Conversation")

##########################

#class definition
class Creatures:
    def __init__(self, x=0, y=0, init_hp=0, max_speed=0, damage=0, init_cal=0, type=0, init_action_to_take=0, angle=90):

        #Hidden
        self.max_speed = max_speed
        self.cal = init_cal
        self.damage = damage

        #Pysical feature that can be observed from outer existence
        self.hp = init_hp
        self.init_hp = init_hp
        self.prev_hp = init_hp
        self.type = type
        self.abs_angle = angle

        self.action = init_action_to_take

        #Observation that this creature observed
        self.observed_state = None

        #World-absolute feature
        self.xCoord = x
        self.yCoord = y
        self.state_info = turtle.Turtle()
        self.agent_visual = turtle.Turtle()
        self.pen_range = turtle.Turtle()

        if self.type == Creature_Name_to_typeId['insects']:
            self.half_range = 7
            value = Map_info(-1, -1, -10)
            self.observed_state = np.full((2*self.half_range + 1, 2*self.half_range + 1), value)
        elif self.type == Creature_Name_to_typeId['mouse']:
            self.x_range_back_0 = 3
            self.x_range_front_0 = 9
            self.y_range_0 = 4
        elif self.type == Creature_Name_to_typeId['deer']:
            self.half_range = 8
            value = Map_info(-1, -1, -10)
            self.observed_state = np.full((2*self.half_range + 1, 2*self.half_range + 1), value)
        elif self.type == Creature_Name_to_typeId['monkey']:
            self.x_range_back_0 = 3
            self.x_range_front_0 = 9
            self.y_range_0 = 6
        elif self.type == Creature_Name_to_typeId['tiger']:
            self.x_range_back_0 = 7
            self.x_range_front_0 = 9
            self.y_range_0 = 5
        elif self.type == Creature_Name_to_typeId['mammoth']:
            self.x_range_back_0 = 2
            self.x_range_front_0 = 10
            self.y_range_0 = 3


    def creature_policy(self):

        # self.action = random.randint(0, 5)

        deadlock_breaker_coin = random.randint(0, 10)
        if deadlock_breaker_coin <2:
            return random.randint(4, 5)
        if self.type == Creature_Name_to_typeId['mammoth']: #"Mammoth"
            self.observe()
            self.action = mammoth_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['tiger']: #"Tiger"
            self.observe()
            self.action = tiger_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['insects']: #"Insects"
            self.observe()
            self.action = insect_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['deer']: #"Deer"
            self.observe()
            self.action = deer_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['mouse']: #"Mouse"
            self.observe()
            self.action = mouse_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['monkey']: #"Monkey"
            self.observe()
            self.action = monkey_policy(observed_state=self.observed_state, self=self)
        elif self.type == Creature_Name_to_typeId['fruit']: #"Fruit"
            self.action = -1 #do nothing
        elif self.type == Creature_Name_to_typeId['p-fruit']: #"P-Fruit"
            self.action = -1 #do nothing

        if self.hp < 0:
            self.__del__()

    def show_info(self):
        type_str = "None"

        if self.type == Creature_Name_to_typeId['mammoth']:
            self.agent_visual.turtlesize(0.5, 1.5)
            wn.colormode(255)
            self.agent_visual.color(210, 105, 30) #brown
            type_str = "Mammoth"
        elif self.type == Creature_Name_to_typeId['tiger']:
            self.agent_visual.turtlesize(0.25, 1)
            self.agent_visual.color("white")
            type_str = "Tiger"
        elif self.type == Creature_Name_to_typeId['insects']:
            self.agent_visual.turtlesize(0.05, 0.1)
            self.agent_visual.color("white")
            type_str = "Insects"
        elif self.type == Creature_Name_to_typeId['deer']:
            self.agent_visual.turtlesize(0.5, 1)
            wn.colormode(255)
            self.agent_visual.color(245,222,179) #light brown
            type_str = "Deer"
        elif self.type == Creature_Name_to_typeId['mouse']:
            self.agent_visual.turtlesize(0.12, 0.3)#(0.06, 0.15)
            wn.colormode(255)
            self.agent_visual.color(169, 169, 169)
            type_str = "Mouse"
        elif self.type == Creature_Name_to_typeId['monkey']:
            self.agent_visual.turtlesize(0.25, 0.5)
            wn.colormode(255)
            self.agent_visual.color(240, 230, 140) #khaki
            type_str = "Monkey"
        elif self.type == Creature_Name_to_typeId['fruit']:
            self.agent_visual.turtlesize(0.10, 0.25)
            self.agent_visual.color("yellow")
            type_str = "Fruit"
        elif self.type == Creature_Name_to_typeId['p-fruit']:
            self.agent_visual.turtlesize(0.10, 0.25)
            self.agent_visual.color("purple")
            type_str = "P-Fruit"

        self.state_info.penup()
        self.state_info.speed(0)
        self.state_info.color("white")
        self.state_info.setposition(w * self.xCoord-width/2, w * self.yCoord-width/2)

        if showing_mode[self.type]:
            infoString = "type: {0}, hp: {1}, quality: {2}".format(type_str, int(self.hp), self.cal)
            self.state_info.write(infoString, font=("Arial", 10, "normal"))

        if self.type == Creature_Name_to_typeId['fruit'] or self.type == Creature_Name_to_typeId['p-fruit']:
            self.agent_visual.shape("circle")
        else:
            self.agent_visual.shape("triangle")
        self.agent_visual.penup()
        self.agent_visual.speed(0)
        self.agent_visual.setposition(w*self.xCoord-width/2, w*self.yCoord-width/2)
        self.agent_visual.setheading(self.abs_angle)


    def clear_img(self):
        self.agent_visual.hideturtle()
        self.agent_visual.clear()
        self.state_info.hideturtle()
        self.state_info.clear()
        self.pen_range.hideturtle()

    def mask_unknown_area_by_obstacle(self, abs_coords_part, search_start_x, search_start_y, search_end_x, search_end_y):

        # if len(TypeList_obst) != len(CoordList_obst):
        #     print("TypeList_obst len: ", len(TypeList_obst))
        #     print("CoordList_obst len: ", len(CoordList_obst))
        #     print("TypeList_obst: ", (TypeList_obst))
        #     print("CoordList_obst: ", (CoordList_obst))

        for i, e in enumerate(TypeList_obst):
            coord = CoordList_obst[i]
            std_x = coord[0]
            std_y = coord[1]
            tos = []

            if e == 0:
                range_ = obstacle_type1_1[0]
            elif e == 1:
                range_ = obstacle_type1_2[0]
            elif e == 2:
                range_ = obstacle_type2_1[1]
            elif e == 3:
                range_ = obstacle_type2_2[1]

            for i in range(range_ + 1):
                if e == 0 or e == 1:
                    x = std_x + i
                    y = std_y
                else:
                    x = std_x
                    y = std_y + i

                if search_start_x <= x and search_end_x >= x \
                        and search_start_y <= y and search_end_y >= y:
                    tos.append((x, y))

                    temp = np.abs(x - self.xCoord) > np.abs(y - self.yCoord)
                    to = None
                    if self.xCoord >= x:
                        if temp:
                            to = (0, y)
                        else:
                            if self.yCoord >= y:
                                to = (x, 0)
                            else:
                                to = (x, coords_width - 1)
                    elif self.xCoord < x:
                        if temp:
                            to = (coords_width - 1, y)
                        else:
                            if self.yCoord >= y:
                                to = (x, 0)
                            else:
                                to = (x, coords_width - 1)
                    tos.append(to)

            if tos:
                cover_start_x = int(tos[0][0])
                cover_start_y = int(tos[0][1])
                cover_end_x = int(tos[0][0])
                cover_end_y = int(tos[0][1])
                for e in tos:
                    if cover_start_x >= e[0]:
                        cover_start_x = int(e[0])
                    if cover_end_x <= e[0]:
                        cover_end_x = int(e[0])
                    if cover_start_y >= e[1]:
                        cover_start_y = int(e[1])
                    if cover_end_y <= e[1]:
                        cover_end_y = int(e[1])

                if cover_start_x < search_start_x:
                    cover_start_x = search_start_x
                if cover_end_x > search_end_x:
                    cover_end_x = search_end_x
                if cover_start_y < search_start_y:
                    cover_start_y = search_start_y
                if cover_end_y > search_end_y:
                    cover_end_y = search_end_y


                for i in range(cover_start_x, cover_end_x + 1):
                    for j in range(cover_start_y, cover_end_y + 1):
                        Pt = abs_coords_part[i, j]
                        abs_coords_part[i, j] = Map_info(Pt.xCoord, Pt.yCoord, -1)
                        if range_render_obst:
                            self.pen_range.penup()
                            self.pen_range.speed(0)
                            self.pen_range.color("yellow")
                            self.pen_range.setposition(i * w - width / 2, j * w - width / 2)
                            self.pen_range.pendown()
                            self.pen_range.circle(1)
                            self.pen_range.penup()

                self.show_range(cover_start_x, cover_start_y, cover_end_x, cover_end_y, range_render_obst, "yellow")


    def show_range(self, cover_start_x, cover_start_y, cover_end_x, cover_end_y, is_render, color):
        if is_render:
            self.pen_range.penup()
            self.pen_range.speed(0)
            self.pen_range.color(color)
            self.pen_range.goto(cover_start_x * w - width / 2, cover_start_y * w - width / 2)
            self.pen_range.pendown()
            self.pen_range.circle(2)
            self.pen_range.goto(self.xCoord * w - width / 2, self.yCoord * w - width / 2)
            self.pen_range.goto(cover_end_x * w - width / 2, cover_end_y * w - width / 2)
            self.pen_range.pendown()
            self.pen_range.circle(2)
            self.pen_range.penup()
            self.pen_range.clear()

    def observe(self):

        if self.type == Creature_Name_to_typeId['insects'] or self.type == Creature_Name_to_typeId['deer']:
            filling_start_x = 0
            filling_start_y = 0
            filling_end_x = 2*self.half_range
            filling_end_y = 2*self.half_range

            search_start_x = self.xCoord - self.half_range
            search_start_y = self.yCoord - self.half_range
            search_end_x = self.xCoord + self.half_range
            search_end_y = self.yCoord + self.half_range

            if search_start_x < 0:
                filling_start_x = -search_start_x
                search_start_x = 0
            if search_start_y < 0:
                filling_start_y = -search_start_y
                search_start_y = 0
            if search_end_x >= coords_width:
                filling_end_x -= (search_end_x - coords_width + 1)
                search_end_x = coords_width - 1
            if search_end_y >= coords_width:
                filling_end_y -= (search_end_y - coords_width + 1)
                search_end_y = coords_width - 1

            search_end_x = int(search_end_x + 1)
            search_end_y = int(search_end_y + 1)
            filling_end_x = int(filling_end_x + 1)
            filling_end_y = int(filling_end_y + 1)

            abs_coords_part = np.copy(abs_coords[:])

            self.mask_unknown_area_by_obstacle(abs_coords_part, search_start_x, search_start_y, search_end_x, search_end_y)
            self.show_range(search_start_x, search_start_y, search_end_x, search_end_y, range_render, "white")

            abs_coords_part = abs_coords_part[search_start_x:search_end_x, search_start_y:search_end_y]
            self.observed_state[filling_start_x:filling_end_x, filling_start_y:filling_end_y] \
                = abs_coords_part





        elif self.type == Creature_Name_to_typeId['mouse'] or self.type == Creature_Name_to_typeId['monkey'] or \
                self.type == Creature_Name_to_typeId['tiger'] or self.type == Creature_Name_to_typeId['mammoth']:

            search_start_x = -self.x_range_back_0
            search_end_x = self.x_range_front_0
            search_start_y = -self.y_range_0
            search_end_y = self.y_range_0

            search_range_0 = np.array([[search_start_x, search_start_y], [search_end_x, search_end_y]])

            value = Map_info(-1, -1, -10)
            self.observed_state = np.full((self.x_range_back_0 + self.x_range_front_0+1, 2*self.y_range_0 + 1), value)

            theta = np.radians(self.abs_angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            search_range_theta = np.dot(search_range_0, R.T)

            search_range_theta[:, 0] += self.xCoord
            search_range_theta[:, 1] += self.yCoord

            min_x_idx = np.argmin(search_range_theta[:, 0])
            min_y_idx = np.argmin(search_range_theta[:, 1])
            max_x_idx = np.argmax(search_range_theta[:, 0])
            max_y_idx = np.argmax(search_range_theta[:, 1])

            min_x = round(search_range_theta[min_x_idx, 0])
            min_y = round(search_range_theta[min_y_idx, 1])

            max_x = round(search_range_theta[max_x_idx, 0] + 1)
            max_y = round(search_range_theta[max_y_idx, 1] + 1)

            new_shape = copy.deepcopy((int(max_x) - int(min_x), int(max_y) - int(min_y)))

            if min_x < 0:
                min_x = 0
            if max_x >= coords_width:
                max_x = coords_width - 1

            if min_y < 0:
                min_y = 0
            if max_y >= coords_width:
                max_y = coords_width - 1
            self.observed_state = np.reshape(self.observed_state, newshape=new_shape)

            x_range = (int(max_x) - int(min_x))
            y_range = (int(max_y) - int(min_y))
            abs_coords_part = np.copy(abs_coords)
            self.mask_unknown_area_by_obstacle(abs_coords_part, int(min_x), int(min_y), int(max_x), int(max_y))
            abs_coords_part = abs_coords_part[int(min_x):int(max_x), int(min_y):int(max_y)]

            self.show_range(int(min_x), int(min_y), int(max_x), int(max_y), range_render, "white")
            if self.abs_angle == 180:
                self.observed_state[new_shape[0]-x_range:new_shape[0], 0:y_range] = \
                    abs_coords_part
            elif self.abs_angle == 90:
                self.observed_state[0:x_range, new_shape[1]-y_range:new_shape[1]] = \
                    abs_coords_part
            else:
                self.observed_state[0:x_range, 0:y_range] = \
                    abs_coords_part




    def __del__(self):
        self.clear_img()

class Map_info:
    def __init__(self, x=0, y=0, type=None):
        self.xCoord = x
        self.yCoord = y
        self.type = type
        self.hp = np.inf
        self.cal = -10


class Agent:
    def __init__(self, x=0, y=0, init_hp=0, max_speed=0, damage=0, init_cal=0, type=9,
                 init_action_to_take=-1, init_com_action_to_take=-1, angle=90, mental_param1=0.5, mental_param2=np.e):

        #Hidden
        self.max_speed = max_speed
        self.cal = init_cal

        self.damage = damage

        #Pysical feature that can be observed from outer existence
        self.hp = init_hp
        self.init_hp = init_hp
        self.prev_hp = init_hp

        self.type = type
        self.abs_angle = angle

        self.action = init_action_to_take
        self.com_action = init_com_action_to_take

        #Observation that this creature observed
        self._observed_state = None
        self.internal_mental_state = 0
        self.internal_mental_state_prev = 0
        self.mental_param1 = mental_param1
        self.mental_param2 = mental_param2
        self.t_felt = 0

        self.expressed_emotion = len(Vocab) - 1 # default for Did not express anything
        self.emotion_intensity = 0 # default for Did not express anything
        self.said_word = len(Emotions) - 1 # default for Did not say anything


        self.is_express_target_in_range = np.zeros((1, Num_Agents), dtype=bool)


        self.who_said_what_to_who = np.full((Num_Agents, Num_Agents), (len(Vocab) - 1))

        self.who_expressed_what_to_who = np.full((Num_Agents, Num_Agents), (len(Emotions) - 1))
        self.who_expressed_how_to_who = np.zeros((Num_Agents, Num_Agents), dtype=int)


        #World-absolute feature
        self.xCoord = x
        self.yCoord = y
        self.state_info = turtle.Turtle()
        self.agent_visual = turtle.Turtle()
        self.pen_range = turtle.Turtle()

        self.x_range_back_0 = 3
        self.x_range_front_0 = 9
        self.y_range_0 = 6

        self.agent_visual = turtle.Turtle()
        self.state_info = turtle.Turtle()


        self.time_frame_w = 10
        self.batch_thold = 100
        self.significancy_thold = 1
        self.added_idx_new = 0

        self.emotion_state1 = np.zeros((4,), dtype=int)
        self.emotion_state2 = np.zeros((4,), dtype=int)
        self.emotion_state3 = np.zeros((4,), dtype=int)
        self.word_state1 = np.zeros((self.time_frame_w, 3), dtype=int)
        self.word_state2 = np.zeros((self.time_frame_w, 3), dtype=int)
        self.word_state3 = np.zeros((self.time_frame_w, 3), dtype=int)

        self.init_words_and_emotions()

        self.said_word_history_stcked_agent1 = []
        self.said_word_history_stcked_agent2 = []
        self.said_word_history_stcked_agent3 = []
        self.emotion_stacked_agent1 = []
        self.emotion_stacked_agent2 = []
        self.emotion_stacked_agent3 = []
        self.reward_stacked = []
        self.reward_stacked_com = []

        self.observed_state_type_stacked = []
        self.observed_state_actions_took_stacked = []
        self.observed_state_hp_stacked = []
        self.observed_state_angle_stacked = []
        self.hf_value_stacked = []



        # self.is_in_range_agent1 = False
        # self.is_in_range_agent2 = False
        # self.is_in_range_agent3 = False
        self.speech_target = np.random.randint(0, Num_Agents)
        self.expression_target = np.random.randint(0, Num_Agents)

        self.f = open('./internal_mental' + str(self.type) + '.csv', 'a', newline='')
        self.makewrite = csv.writer(self.f)

    def init_words_and_emotions(self):

        target = -1

        self.emotion_state1[0] = len(Emotions) - 1
        self.emotion_state1[1] = 0
        self.emotion_state1[2] = target
        self.emotion_state1[3] = 1  # hf value
        self.word_state1[0, 0] = len(Vocab) - 1
        self.word_state1[0, 1] = target
        self.word_state1[0, 2] = 1

        self.emotion_state2[0] = len(Emotions) - 1
        self.emotion_state2[1] = 0
        self.emotion_state2[2] = target
        self.emotion_state2[3] = 1  # hf value
        self.word_state2[0, 0] = len(Vocab) - 1
        self.word_state2[0, 1] = target
        self.word_state2[0, 2] = 1

        self.emotion_state3[0] = len(Emotions) - 1
        self.emotion_state3[1] = 0
        self.emotion_state3[2] = target
        self.emotion_state3[3] = 1  # hf value
        self.word_state3[0, 0] = len(Vocab) - 1
        self.word_state3[0, 1] = target
        self.word_state3[0, 2] = 1

        self.expression_target = -1
        self.speech_target = -1

        # self.is_in_range_agent1 = False
        # self.is_in_range_agent2 = False
        # self.is_in_range_agent3 = False

    def is_in_range_agent_sound(self, i, j):

        agent_from = False
        agent_to = False

        if len(AgentList) == Num_Agents:
            agent_from = AgentList[i]
            agent_to = AgentList[j]
        else:
            for a in AgentList:
                idx = a.type - Creature_Name_to_typeId['agent1']
                if idx == i:
                    agent_from = a
                elif idx == j:
                    agent_to = a

        if (not agent_to) or (not agent_from):
            return False

        manhattan_dis = abs(agent_from.xCoord - agent_to.xCoord) + abs(agent_from.yCoord - agent_to.yCoord)

        # if self.type == Creature_Name_to_typeId['agent1']:
        #     print("dis: {}, i: {}, j: {}, isin?: {}".format(manhattan_dis, i, j, (manhattan_dis <= Talk_range)))

        if manhattan_dis <= Talk_range:
            return True
        return False

    def fillout_emotion_word_single_state(self):

        word_state1 = np.full((self.time_frame_w, 3), (len(Vocab) - 1))
        word_state2 = np.full((self.time_frame_w, 3), (len(Vocab) - 1))
        word_state3 = np.full((self.time_frame_w, 3), (len(Vocab) - 1))

        word_state1[1:self.time_frame_w, :] = self.word_state1[0:self.time_frame_w-1, :]
        word_state2[1:self.time_frame_w, :] = self.word_state2[0:self.time_frame_w-1, :]
        word_state3[1:self.time_frame_w, :] = self.word_state3[0:self.time_frame_w-1, :]

        #print(self.com_action)
        #print("speech: {}, expression: {}".format(self.speech_target, self.expression_target))

        if not self.is_in_range_agent_sound(self.type - Creature_Name_to_typeId['agent1'], self.speech_target):
            self.speech_target = -1
        #print("exp t: ", self.expression_target)
        if not self.is_express_target_in_range[0, self.expression_target]:
            self.expression_target = -1

        for i in range(Num_Agents):
            for j in range(Num_Agents):
                expressed_emotion_from_i_to_j = self.who_expressed_what_to_who[i, j]
                expressed_emotion_intensity_from_i_to_j = self.who_expressed_how_to_who[i, j]
                said_word_from_i_to_j = self.who_said_what_to_who[i, j]
                if i == 0:
                    if self.is_express_target_in_range[0, i] and self.is_express_target_in_range[0, j]:
                        self.emotion_state1[0] = expressed_emotion_from_i_to_j
                        self.emotion_state1[1] = expressed_emotion_intensity_from_i_to_j
                        self.emotion_state1[2] = j
                        self.emotion_state1[3] = 1  # hf value

                    if self.is_in_range_agent_sound(self.type - Creature_Name_to_typeId['agent1'], i):
                        word_state1[0, 0] = said_word_from_i_to_j
                        word_state1[0, 1] = j
                        word_state1[0, 2] = 1

                elif i == 1:
                    if self.is_express_target_in_range[0, i] and self.is_express_target_in_range[0, j]:
                        self.emotion_state2[0] = expressed_emotion_from_i_to_j
                        self.emotion_state2[1] = expressed_emotion_intensity_from_i_to_j
                        self.emotion_state2[2] = j
                        self.emotion_state2[3] = 1  # hf value

                    if self.is_in_range_agent_sound(self.type - Creature_Name_to_typeId['agent1'], i):
                        word_state2[0, 0] = said_word_from_i_to_j
                        word_state2[0, 1] = j
                        word_state2[0, 2] = 1

                elif i == 2:
                    if self.is_express_target_in_range[0, i] and self.is_express_target_in_range[0, j]:
                        self.emotion_state3[0] = expressed_emotion_from_i_to_j
                        self.emotion_state3[1] = expressed_emotion_intensity_from_i_to_j
                        self.emotion_state3[2] = j
                        self.emotion_state3[3] = 1  # hf value

                    if self.is_in_range_agent_sound(self.type - Creature_Name_to_typeId['agent1'], i):
                        word_state3[0, 0] = said_word_from_i_to_j
                        word_state3[0, 1] = j
                        word_state3[0, 2] = 1


        self.word_state1 = word_state1
        self.word_state2 = word_state2
        self.word_state3 = word_state3


    def agent_policy(self, time_step=0, policy_learner=None):

        # if self.internal_mental_state != 0:
        #     self.apply_own_emotion()

        self.observe_state()
        self.time_step = time_step
        self.fillout_emotion_word_single_state()

        self.hf_value = 1

        idx = self.type - Creature_Name_to_typeId['agent1']
        if len(observed_state_type_stacked[idx]) < 2000:
            observed_state_type_stacked[idx].append([self.observed_state_type])
            observed_state_angle_stacked[idx].append([self.observed_state_angle])
            observed_state_hp_stacked[idx].append([self.observed_state_hp])
            observed_state_actions_took_stacked[idx].append([self.observed_state_actions_took])
            said_word_history_stcked_agent1[idx].append([self.word_state1])
            said_word_history_stcked_agent2[idx].append([self.word_state2])
            said_word_history_stcked_agent3[idx].append([self.word_state3])
            emotion_stacked_agent1[idx].append([self.emotion_state1])
            emotion_stacked_agent2[idx].append([self.emotion_state2])
            emotion_stacked_agent3[idx].append([self.emotion_state3])
            hf_value_stacked[idx].append([self.hf_value])
            self.added_idx_new = len(observed_state_type_stacked[idx]) - 1
        else:
            replaceind = np.random.randint(0, 2000)
            observed_state_type_stacked[idx][replaceind] = ([self.observed_state_type])
            observed_state_angle_stacked[idx][replaceind] = ([self.observed_state_angle])
            observed_state_hp_stacked[idx][replaceind] = ([self.observed_state_hp])
            observed_state_actions_took_stacked[idx][replaceind] = ([self.observed_state_actions_took])
            said_word_history_stcked_agent1[idx][replaceind] = ([self.word_state1])
            said_word_history_stcked_agent2[idx][replaceind] = ([self.word_state2])
            said_word_history_stcked_agent3[idx][replaceind] = ([self.word_state3])
            emotion_stacked_agent1[idx][replaceind] = ([self.emotion_state1])
            emotion_stacked_agent2[idx][replaceind] = ([self.emotion_state2])
            emotion_stacked_agent3[idx][replaceind] = ([self.emotion_state3])
            hf_value_stacked[idx][replaceind] = ([self.hf_value])
            self.added_idx_new = replaceind


        param1 = np.vstack([[self.observed_state_type]])
        param2 = np.vstack([[self.observed_state_hp]])
        param3 = np.vstack([[self.observed_state_angle]])
        param4 = np.vstack([[self.observed_state_actions_took]])
        param5 = np.vstack([[self.hf_value]])
        param6 = np.vstack([[self.word_state1]])
        param7 = np.vstack([[self.word_state2]])
        param8 = np.vstack([[self.word_state3]])
        param9 = np.vstack([[self.emotion_state1]])
        param10 = np.vstack([[self.emotion_state2]])
        param11 = np.vstack([[self.emotion_state3]])

        #print("time step: ", self.time_step)
        if self.type == Creature_Name_to_typeId["agent1"]: #"Agnet1"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express

            self.action, self.com_action = agent_policy1(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=False)
        elif self.type == Creature_Name_to_typeId["agent2"]: #"Agnet2"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express
            self.action, self.com_action = agent_policy2(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=False)
        elif self.type == Creature_Name_to_typeId["agent3"]: #"Agnet3"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express
            self.action, self.com_action = agent_policy3(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=False)
        # if self.type == Creature_Name_to_typeId['agent1']:#self.said_word < len(Vocab):
        #     print("self.said_word: ", self.said_word)




    def train(self, is_terminal, policy_learner=None):

        #observe the im reward
        hps = self.hp #+= self.hp
        prev_hps = self.prev_hp#+= self.prev_hp
        # internal_mental = self.internal_mental_state
        internal_mental = self.internal_mental_state #+= self.internal_mental_state
        self.reward = hps - prev_hps #+ internal_mental[0, 0]/10
        #print(reward)

        reward = 1/(1 + np.exp(-self.reward))
        self.hf_value = 1

        idx = self.type - Creature_Name_to_typeId['agent1']
        significant_idx_new = 0
        if self.reward >= self.significancy_thold:
            if len(observed_state_type_stacked_big[idx]) < 2000:
                observed_state_type_stacked_big[idx].append([self.observed_state_type])
                observed_state_angle_stacked_big[idx].append([self.observed_state_angle])
                observed_state_hp_stacked_big[idx].append([self.observed_state_hp])
                observed_state_actions_took_stacked_big[idx].append([self.observed_state_actions_took])
                said_word_history_stcked_agent1_big[idx].append([self.word_state1])
                said_word_history_stcked_agent2_big[idx].append([self.word_state2])
                said_word_history_stcked_agent3_big[idx].append([self.word_state3])
                emotion_stacked_agent1_big[idx].append([self.emotion_state1])
                emotion_stacked_agent2_big[idx].append([self.emotion_state2])
                emotion_stacked_agent3_big[idx].append([self.emotion_state3])
                hf_value_stacked_big[idx].append([self.hf_value])
                significant_idx_new = len(hf_value_stacked_big[idx]) - 1
            else:
                replaceind = np.random.randint(0, 2000)
                observed_state_type_stacked_big[idx][replaceind] = ([self.observed_state_type])
                observed_state_angle_stacked_big[idx][replaceind] = ([self.observed_state_angle])
                observed_state_hp_stacked_big[idx][replaceind] = ([self.observed_state_hp])
                observed_state_actions_took_stacked_big[idx][replaceind] = ([self.observed_state_actions_took])
                said_word_history_stcked_agent1_big[idx][replaceind] = ([self.word_state1])
                said_word_history_stcked_agent2_big[idx][replaceind] = ([self.word_state2])
                said_word_history_stcked_agent3_big[idx][replaceind] = ([self.word_state3])
                emotion_stacked_agent1_big[idx][replaceind] = ([self.emotion_state1])
                emotion_stacked_agent2_big[idx][replaceind] = ([self.emotion_state2])
                emotion_stacked_agent3_big[idx][replaceind] = ([self.emotion_state3])
                hf_value_stacked_big[idx][replaceind] = ([self.hf_value])
                significant_idx_new = replaceind

        if self.internal_mental_state != 0:
            self.apply_own_emotion()

        #observe excuted state
        self.observe_state()
        self.fillout_emotion_word_single_state()

        self.hf_value = 1

        param1 = np.vstack([[self.observed_state_type]])
        param2 = np.vstack([[self.observed_state_hp]])
        param3 = np.vstack([[self.observed_state_angle]])
        param4 = np.vstack([[self.observed_state_actions_took]])
        param5 = np.vstack([[self.hf_value]])
        param6 = np.vstack([[self.word_state1]])
        param7 = np.vstack([[self.word_state2]])
        param8 = np.vstack([[self.word_state3]])
        param9 = np.vstack([[self.emotion_state1]])
        param10 = np.vstack([[self.emotion_state2]])
        param11 = np.vstack([[self.emotion_state3]])

        action_pred_value = 0
        com_action_pred_value = 0

        if self.type == Creature_Name_to_typeId["agent1"]: #"Agnet1"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express

            action_pred_value, com_action_pred_value = agent_policy1(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=True)
        elif self.type == Creature_Name_to_typeId["agent2"]: #"Agnet2"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express
            action_pred_value, com_action_pred_value = agent_policy2(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=True)
        elif self.type == Creature_Name_to_typeId["agent3"]: #"Agnet3"
            #self.init_learner()
            #the function modifies the word to say, and emotion to express
            action_pred_value, com_action_pred_value = agent_policy3(type_input=param1,
                                                         hp_input=param2,
                                                         angle_input=param3,
                                                         action_took_input=param4,
                                                         hf_value=param5, words_input1=param6,
                                                         words_input2=param7,
                                                         words_input3=param8,
                                                         emotion_input1=param9,
                                                         emotion_input2=param10,
                                                         emotion_input3=param11, learner=policy_learner,
                                                                             self=self, is_pred_mode=True)

        if not is_terminal:
            reward = reward + action_pred_value

        if len(reward_stacked) < 2000:
            reward_stacked[idx].append([reward])
            reward_stacked_com[idx].append([reward])
        else:
            reward_stacked[idx][self.added_idx_new] = ([reward])
            reward_stacked_com[idx][self.added_idx_new] = ([reward])

        if self.reward >= self.significancy_thold:
            if len(reward_stacked_big) < 2000:
                reward_stacked_big[idx].append([reward])
                reward_stacked_com_big[idx].append([reward])
            else:
                reward_stacked_big[idx][significant_idx_new] = ([reward])
                reward_stacked_com_big[idx][significant_idx_new] = ([reward])


        if (self.time_step % self.batch_thold == 0 and self.time_step != 0) or is_terminal:
            print("trained a batch!")
            sampling_pop = self.batch_thold - 1
            sampling_pop2 = self.batch_thold - 1
            list_length = len(observed_state_type_stacked[idx])
            if self.batch_thold-1 >= list_length:
                sampling_pop = list_length
            list_length = len(observed_state_type_stacked_big[idx])
            if self.batch_thold-1 >= list_length:
                sampling_pop2 = list_length

            t = random.sample(range(len(observed_state_type_stacked[idx])), sampling_pop)

            patch_in_type = list(itemgetter(*t)(observed_state_type_stacked[idx]))
            patch_in_actions = list(itemgetter(*t)(observed_state_actions_took_stacked[idx]))
            patch_in_hp = list(itemgetter(*t)(observed_state_hp_stacked[idx]))
            patch_in_angle = list(itemgetter(*t)(observed_state_angle_stacked[idx]))
            words_input1 = list(itemgetter(*t)(said_word_history_stcked_agent1[idx]))
            words_input2 = list(itemgetter(*t)(said_word_history_stcked_agent2[idx]))
            words_input3 = list(itemgetter(*t)(said_word_history_stcked_agent3[idx]))
            emotion_input1 = list(itemgetter(*t)(emotion_stacked_agent1[idx]))
            emotion_input2 = list(itemgetter(*t)(emotion_stacked_agent2[idx]))
            emotion_input3 = list(itemgetter(*t)(emotion_stacked_agent3[idx]))
            reward_patch = list(itemgetter(*t)(reward_stacked[idx]))
            reward_patch_com = list(itemgetter(*t)(reward_stacked_com[idx]))
            hf_value = list(itemgetter(*t)(hf_value_stacked[idx]))

            t2 = random.sample(range(len(observed_state_type_stacked_big[idx])), sampling_pop2)
            if len(observed_state_type_stacked_big[idx]) > 1:
                #print("observed_state_type_stacked_big[idx]) > 0")
                patch_in_type_big = list(itemgetter(*t2)(observed_state_type_stacked_big[idx]))
                patch_in_actions_big = list(itemgetter(*t2)(observed_state_actions_took_stacked_big[idx]))
                patch_in_hp_big = list(itemgetter(*t2)(observed_state_hp_stacked_big[idx]))
                patch_in_angle_big = list(itemgetter(*t2)(observed_state_angle_stacked_big[idx]))
                words_input1_big = list(itemgetter(*t2)(said_word_history_stcked_agent1_big[idx]))
                words_input2_big = list(itemgetter(*t2)(said_word_history_stcked_agent2_big[idx]))
                words_input3_big = list(itemgetter(*t2)(said_word_history_stcked_agent3_big[idx]))
                emotion_input1_big = list(itemgetter(*t2)(emotion_stacked_agent1_big[idx]))
                emotion_input2_big = list(itemgetter(*t2)(emotion_stacked_agent2_big[idx]))
                emotion_input3_big = list(itemgetter(*t2)(emotion_stacked_agent3_big[idx]))
                reward_patch_big = list(itemgetter(*t2)(reward_stacked_big[idx]))
                reward_patch_com_big = list(itemgetter(*t2)(reward_stacked_com_big[idx]))
                hf_value_big = list(itemgetter(*t2)(hf_value_stacked_big[idx]))
            else:
                #print("observed_state_type_stacked_big[idx]) <= 0")
                t = random.sample(range(len(observed_state_type_stacked[idx])), sampling_pop)
                patch_in_type_big = list(itemgetter(*t)(observed_state_type_stacked[idx]))
                patch_in_actions_big = list(itemgetter(*t)(observed_state_actions_took_stacked[idx]))
                patch_in_hp_big = list(itemgetter(*t)(observed_state_hp_stacked[idx]))
                patch_in_angle_big = list(itemgetter(*t)(observed_state_angle_stacked[idx]))
                words_input1_big = list(itemgetter(*t)(said_word_history_stcked_agent1[idx]))
                words_input2_big = list(itemgetter(*t)(said_word_history_stcked_agent2[idx]))
                words_input3_big = list(itemgetter(*t)(said_word_history_stcked_agent3[idx]))
                emotion_input1_big = list(itemgetter(*t)(emotion_stacked_agent1[idx]))
                emotion_input2_big = list(itemgetter(*t)(emotion_stacked_agent2[idx]))
                emotion_input3_big = list(itemgetter(*t)(emotion_stacked_agent3[idx]))
                reward_patch_big = list(itemgetter(*t)(reward_stacked[idx]))
                reward_patch_com_big = list(itemgetter(*t)(reward_stacked_com[idx]))
                hf_value_big = list(itemgetter(*t)(hf_value_stacked[idx]))

            # print("patch_in_type: ", patch_in_type)
            #print("patch_in_type_big: ", patch_in_type_big)
            patch_in_type = np.vstack(patch_in_type + patch_in_type_big)
            patch_in_actions = np.vstack(patch_in_actions + patch_in_actions_big)
            patch_in_hp = np.vstack(patch_in_hp + patch_in_hp_big)
            patch_in_angle = np.vstack(patch_in_angle + patch_in_angle_big)
            words_input1 = np.vstack(words_input1 + words_input1_big)
            words_input2 = np.vstack(words_input2 + words_input2_big)
            words_input3 = np.vstack(words_input3 + words_input3_big)
            emotion_input1 = np.vstack(emotion_input1 + emotion_input1_big)
            emotion_input2 = np.vstack(emotion_input2 + emotion_input2_big)
            emotion_input3 = np.vstack(emotion_input3 + emotion_input3_big)
            reward_patch = np.vstack(reward_patch + reward_patch_big)
            reward_patch_com = np.vstack(reward_patch_com + reward_patch_com_big)
            hf_value = np.vstack(hf_value + hf_value_big)
            #
            # print("patch_in_type: ", patch_in_type.shape)
            # print("patch_in_actions: ", patch_in_actions.shape)
            # print("patch_in_hp: ", patch_in_hp.shape)
            # print("patch_in_angle: ", patch_in_angle.shape)
            # print("words_input1: ", words_input1.shape)
            # print("words_input2: ", words_input2.shape)
            # print("words_input3: ", words_input3.shape)
            # print("emotion_input1: ", emotion_input1.shape)
            # print("emotion_input2: ", emotion_input2.shape)
            # print("emotion_input3: ", emotion_input3.shape)
            # print("reward_patch: ", reward_patch.shape)
            # print("reward_patch_com: ", reward_patch_com.shape)
            # print("hf_value: ", hf_value.shape)
            #
            # print("l1: ", len(observed_state_type_stacked))
            # print("l2: ", len(observed_state_actions_took_stacked))
            # print("l3: ", len(observed_state_hp_stacked_big[idx]))
            # print("l4: ", len(observed_state_angle_stacked_big[idx]))
            # print("l12: ", len(reward_stacked_big[idx]))
            # print("l13: ", len(reward_stacked_com_big[idx]))



            lr = 1e-6
            #lr /= (1 + math.log10(self.time_step))
            if self.type == Creature_Name_to_typeId['agent1']:
                policy_learner.train_batch(gt_reward=reward_patch, type_input=patch_in_type, hp_input=patch_in_hp,
                                           angle_input=patch_in_angle, action_took_input=patch_in_actions,
                                           hf_value=hf_value, words_input1=words_input1,
                                           words_input2=words_input2, words_input3=words_input3,
                                           emotion_input1=emotion_input1, emotion_input2=emotion_input2,
                                           emotion_input3=emotion_input3, is_training=True, learning_rate_shifting=lr
                                           , com_reward=reward_patch_com)
            elif self.type == Creature_Name_to_typeId['agent2']:
                policy_learner.train_batch2(gt_reward=reward_patch, type_input=patch_in_type, hp_input=patch_in_hp,
                                           angle_input=patch_in_angle, action_took_input=patch_in_actions,
                                           hf_value=hf_value, words_input1=words_input1,
                                           words_input2=words_input2, words_input3=words_input3,
                                           emotion_input1=emotion_input1, emotion_input2=emotion_input2,
                                           emotion_input3=emotion_input3, is_training=True, learning_rate_shifting=lr
                                            , com_reward=reward_patch_com)
            elif self.type == Creature_Name_to_typeId['agent3']:
                policy_learner.train_batch3(gt_reward=reward_patch, type_input=patch_in_type, hp_input=patch_in_hp,
                                           angle_input=patch_in_angle, action_took_input=patch_in_actions,
                                           hf_value=hf_value, words_input1=words_input1,
                                           words_input2=words_input2, words_input3=words_input3,
                                           emotion_input1=emotion_input1, emotion_input2=emotion_input2,
                                           emotion_input3=emotion_input3, is_training=True, learning_rate_shifting=lr
                                            , com_reward=reward_patch_com)


            #reset stacked arrays
            # if self.time_step % (20*self.batch_thold) == 0 and self.type == Creature_Name_to_typeId['agent1']:
            #     policy_learner.saver.save(policy_learner.sess, './model/agent_policy_model' + str(self.time_step))
            #     print("saved updated model!")

            # self.observed_state_type_stacked = []
            # self.observed_state_actions_took_stacked = []
            # self.observed_state_hp_stacked = []
            # self.observed_state_angle_stacked = []
            # self.said_word_history_stcked_agent1 = []
            # self.said_word_history_stcked_agent2 = []
            # self.said_word_history_stcked_agent3 = []
            # self.emotion_stacked_agent1 = []
            # self.emotion_stacked_agent2 = []
            # self.emotion_stacked_agent3 = []
            # self.reward_stacked = []
            # self.reward_stacked_com = []
            # self.hf_value_stacked = []





    def init_learner(self):

        global is_learner_set
        global policy_learner

        if not is_learner_set:
            policy_learner = agent_net(physical_observation_shape=self.observed_state_type.shape,
                                             should_load=should_load_, length_seq_w=self.time_frame_w)
            is_learner_set = True

    def apply_own_emotion(self):
        self.t_felt += 1
        if np.abs(self.internal_mental_state_prev) < np.abs(self.internal_mental_state) or \
                (self.internal_mental_state *self.internal_mental_state_prev < 0):
            self.t_felt = 0
        self.internal_mental_state_prev = copy.deepcopy(self.internal_mental_state)
        self.internal_mental_state = self.mental_param1 * \
                                     self.internal_mental_state*pow(self.mental_param2, -self.t_felt)
        #self.hp += self.internal_mental_state
        self.makewrite.writerow([self.time_step, self.internal_mental_state[0, 0]])


    def mask_unknown_area_by_obstacle(self, abs_coords_part, search_start_x, search_start_y, search_end_x, search_end_y):

        for i, e in enumerate(TypeList_obst):
            coord = CoordList_obst[i]
            std_x = coord[0]
            std_y = coord[1]
            tos = []

            range_ = -1
            if e == 0:
                range_ = obstacle_type1_1[0]
            elif e == 1:
                range_ = obstacle_type1_2[0]
            elif e == 2:
                range_ = obstacle_type2_1[1]
            elif e == 3:
                range_ = obstacle_type2_2[1]

            for i in range(range_ + 1):
                if e == 0 or e == 1:
                    x = std_x + i
                    y = std_y
                else:
                    x = std_x
                    y = std_y + i

                if search_start_x <= x and search_end_x >= x \
                        and search_start_y <= y and search_end_y >= y:
                    tos.append((x, y))

                    temp = np.abs(x - self.xCoord) > np.abs(y - self.yCoord)
                    to = None
                    if self.xCoord >= x:
                        if temp:
                            to = (0, y)
                        else:
                            if self.yCoord >= y:
                                to = (x, 0)
                            else:
                                to = (x, coords_width - 1)
                    elif self.xCoord < x:
                        if temp:
                            to = (coords_width - 1, y)
                        else:
                            if self.yCoord >= y:
                                to = (x, 0)
                            else:
                                to = (x, coords_width - 1)
                    tos.append(to)

            if tos:
                cover_start_x = int(tos[0][0])
                cover_start_y = int(tos[0][1])
                cover_end_x = int(tos[0][0])
                cover_end_y = int(tos[0][1])
                for e in tos:
                    if cover_start_x >= e[0]:
                        cover_start_x = int(e[0])
                    if cover_end_x <= e[0]:
                        cover_end_x = int(e[0])
                    if cover_start_y >= e[1]:
                        cover_start_y = int(e[1])
                    if cover_end_y <= e[1]:
                        cover_end_y = int(e[1])

                if cover_start_x < search_start_x:
                    cover_start_x = search_start_x
                if cover_end_x > search_end_x:
                    cover_end_x = search_end_x
                if cover_start_y < search_start_y:
                    cover_start_y = search_start_y
                if cover_end_y > search_end_y:
                    cover_end_y = search_end_y


                for i in range(cover_start_x, cover_end_x + 1):
                    for j in range(cover_start_y, cover_end_y + 1):
                        Pt = abs_coords_part[i, j]
                        abs_coords_part[i, j] = Map_info(Pt.xCoord, Pt.yCoord, -1)
                        if range_render_obst:
                            self.pen_range.penup()
                            self.pen_range.speed(0)
                            self.pen_range.color("yellow")
                            self.pen_range.setposition(i * w - width / 2, j * w - width / 2)
                            self.pen_range.pendown()
                            self.pen_range.circle(1)
                            self.pen_range.penup()

                self.show_range(cover_start_x, cover_start_y, cover_end_x, cover_end_y, range_render_obst, "yellow")


    def show_range(self, cover_start_x, cover_start_y, cover_end_x, cover_end_y, is_render, color):
        if is_render:
            self.pen_range.penup()
            self.pen_range.speed(0)
            self.pen_range.color(color)
            self.pen_range.goto(cover_start_x * w - width / 2, cover_start_y * w - width / 2)
            self.pen_range.pendown()
            self.pen_range.circle(2)
            self.pen_range.goto(self.xCoord * w - width / 2, self.yCoord * w - width / 2)
            self.pen_range.goto(cover_end_x * w - width / 2, cover_end_y * w - width / 2)
            self.pen_range.pendown()
            self.pen_range.circle(2)
            self.pen_range.penup()
            self.pen_range.clear()

    def observe_state(self):
        search_start_x = -self.x_range_back_0
        search_end_x = self.x_range_front_0
        search_start_y = -self.y_range_0
        search_end_y = self.y_range_0

        search_range_0 = np.array([[search_start_x, search_start_y], [search_end_x, search_end_y]])

        value = Map_info(-1, -1, -10)
        self._observed_state = np.full((self.x_range_back_0 + self.x_range_front_0 + 1, 2 * self.y_range_0 + 1), value)

        theta = np.radians(self.abs_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        search_range_theta = np.dot(search_range_0, R.T)

        search_range_theta[:, 0] += self.xCoord
        search_range_theta[:, 1] += self.yCoord

        min_x_idx = np.argmin(search_range_theta[:, 0])
        min_y_idx = np.argmin(search_range_theta[:, 1])
        max_x_idx = np.argmax(search_range_theta[:, 0])
        max_y_idx = np.argmax(search_range_theta[:, 1])

        min_x = round(search_range_theta[min_x_idx, 0])
        min_y = round(search_range_theta[min_y_idx, 1])

        max_x = round(search_range_theta[max_x_idx, 0] + 1)
        max_y = round(search_range_theta[max_y_idx, 1] + 1)

        new_shape = copy.deepcopy((int(max_x) - int(min_x), int(max_y) - int(min_y)))

        if min_x < 0:
            min_x = 0
        if max_x >= coords_width:
            max_x = coords_width - 1

        if min_y < 0:
            min_y = 0
        if max_y >= coords_width:
            max_y = coords_width - 1
        self._observed_state = np.reshape(self._observed_state, newshape=new_shape)

        x_range = (int(max_x) - int(min_x))
        y_range = (int(max_y) - int(min_y))
        abs_coords_part = np.copy(abs_coords)
        self.mask_unknown_area_by_obstacle(abs_coords_part, int(min_x), int(min_y), int(max_x), int(max_y))
        abs_coords_part = abs_coords_part[int(min_x):int(max_x), int(min_y):int(max_y)]

        self.show_range(int(min_x), int(min_y), int(max_x), int(max_y), range_render, "white")
        if self.abs_angle == 180:
            self._observed_state[new_shape[0] - x_range:new_shape[0], 0:y_range] = \
                abs_coords_part
        elif self.abs_angle == 90:
            self._observed_state[0:x_range, new_shape[1] - y_range:new_shape[1]] = \
                abs_coords_part
        else:
            self._observed_state[0:x_range, 0:y_range] = \
                abs_coords_part


        self.observed_state_type = np.zeros(new_shape, dtype=float)
        self.observed_state_actions_took = np.zeros(new_shape, dtype=float)
        self.observed_state_hp = np.zeros(new_shape, dtype=float)
        self.observed_state_angle = np.zeros(new_shape, dtype=float)
        self.is_express_target_in_range = np.zeros((1, Num_Agents), dtype=bool)

        # self.words_spoken_toward_me = np.zeros((1, Num_Agents), dtype=int)
        # self.who_said_what_to_who = np.zeros((Num_Agents, Num_Agents), dtype=int)
        #
        # self.emotion_expressed_toward_me = np.zeros((1, Num_Agents), dtype=int)
        # self.who_expressed_what_to_who = np.zeros((Num_Agents, Num_Agents), dtype=int)
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                point = self._observed_state[i, j]
                self.observed_state_type[i, j] = point.type
                # if point.type == Creature_Name_to_typeId['agent1']:
                #     self.is_in_range_agent1 = True
                # elif point.type == Creature_Name_to_typeId['agent2']:
                #     self.is_in_range_agent2 = True
                # elif point.type == Creature_Name_to_typeId['agent3']:
                #     self.is_in_range_agent3 = True

                if point.type > 0:
                    self.observed_state_actions_took[i, j] = point.action
                    self.observed_state_hp[i, j] = point.hp
                    if point.abs_angle != -90:
                        self.observed_state_angle[i, j] = point.abs_angle
                    else:
                        self.observed_state_angle[i, j] = 270

                    if point.type >= Creature_Name_to_typeId['agent1'] and \
                            point.type <= Creature_Name_to_typeId['agent'+ str(Num_Agents)]:
                        self.is_express_target_in_range[0, point.type - Creature_Name_to_typeId['agent1']] = True
                else:
                    self.observed_state_actions_took[i, j] = -2
                    self.observed_state_hp[i, j] = -2
                    self.observed_state_angle[i, j] = -1


    def show_info(self):

        type_str = "None"

        # target_x = -1
        # target_y = -1
        # es = None
        ws = None

        if self.type == Creature_Name_to_typeId['agent1']:
            type_str = "Agent A"
            ws = self.word_state1
            # es = self.emotion_state1
        elif self.type == Creature_Name_to_typeId['agent2']:
            type_str = "Agent B"
            ws = self.word_state2
            # es = self.emotion_state2
        elif self.type == Creature_Name_to_typeId['agent3']:
            type_str = "Agent C"
            ws = self.word_state3
            # es = self.emotion_state3

        self.agent_visual.turtlesize(0.5, 1)

        self.state_info.penup()
        self.state_info.speed(0)
        self.state_info.color("white")
        self.state_info.setposition(w * self.xCoord-width/2, w * self.yCoord-width/2)

        if showing_mode[self.type]:
            infoString = "type: {0}, hp: {1}, quality: {2}".format(type_str, int(self.hp), self.cal)

            self.state_info.write(infoString, font=("Arial", 10, "normal"))
            self.state_info.penup()
            # show the expressions
            self.state_info.setposition(w * self.xCoord - width / 2, w * self.yCoord - width / 2 - 20)
            # self.state_info.setposition(w * self.xCoord-width/2, w * self.yCoord-width/2)

            said_seq = ''
            for i, e in enumerate(ws[:, 0]):
                said_seq += (Vocab[e])
            infoString = "said: {0} to: {1}, expressed: {2} to: {3}, pa: {4}".format(said_seq, self.speech_target,
                                                                            self.expressed_emotion,
                                                                            self.expression_target, self.action)
            self.state_info.write(infoString, font=("Arial", 10, "normal"))
        self.state_info.penup()

        if self.type == Creature_Name_to_typeId['agent1']:
            self.state_info.color("red")
        elif self.type == Creature_Name_to_typeId['agent2']:
            self.state_info.color("yellow")
        elif self.type == Creature_Name_to_typeId['agent3']:
            self.state_info.color("green")

        for a in AgentList:
            # if self.type == Creature_Name_to_typeId['agent1']:
            if self.speech_target >= 0 and a.type - Creature_Name_to_typeId['agent1'] == self.speech_target:
                self.state_info.setposition(w * self.xCoord - width / 2 - 6, w * self.yCoord - width / 2 - 20)
                self.state_info.pensize(1)
                self.state_info.pendown()
                self.state_info.goto(w * a.xCoord - width / 2, w * a.yCoord - width / 2)
                self.state_info.hideturtle()
                self.state_info.penup()
            if self.expression_target >= 0 and a.type - Creature_Name_to_typeId['agent1'] == self.expression_target:
                self.state_info.penup()
                # self.state_info.setposition(w * self.xCoord - width / 2 + 30, w * self.yCoord - width / 2 - 20)
                self.state_info.setposition(w * self.xCoord - width / 2, w * self.yCoord - width / 2)
                self.state_info.pensize(3)
                self.state_info.pendown()
                self.state_info.goto(w * a.xCoord - width / 2, w * a.yCoord - width / 2)
                self.state_info.penup()
                self.state_info.hideturtle()

        self.agent_visual.color("yellow")
        self.agent_visual.shape("triangle")
        self.agent_visual.penup()
        self.agent_visual.speed(0)
        self.agent_visual.setposition(w*self.xCoord-width/2, w*self.yCoord-width/2)
        self.agent_visual.setheading(self.abs_angle)



    def record_state(self):

        f = open("transition"+str(self.type)+".txt", 'a', encoding='utf-8')
        f.write(str(self.xCoord)+" "+str(self.yCoord)+'\n')
        f.close()
        return

    def clear_img(self):
        self.agent_visual.hideturtle()
        self.agent_visual.clear()
        self.state_info.hideturtle()
        self.state_info.clear()

    def reset_interaction_arrays(self):
        #Update em at the next observation and action selection
        # self.expressed_emotion = len(Emotions) - 1  # -1 for Did not say anything
        # self.emotion_intensity = 0
        # self.said_word = len(Vocab) - 1  # -1 for Did not say anything

        #self.is_express_target_in_range = np.zeros((1, Num_Agents), dtype=bool)

        self.who_said_what_to_who = np.full((Num_Agents, Num_Agents), (len(Vocab) - 1))
        self.who_expressed_what_to_who = np.full((Num_Agents, Num_Agents), (len(Vocab) - 1))
        self.who_expressed_how_to_who = np.zeros((Num_Agents, Num_Agents), dtype=int)

        #self.speech_target = -1
        #self.expression_target = -1
        #self.init_words_and_emotions()


    def __del__(self):
        self.clear_img()
        self.makewrite.writerow([self.time_step, -1])
        self.f.close()


class Human:
    def __init__(self, x=0, y=0, init_hp=0, max_speed=0, damage=0, init_cal=0, type=12,
                 init_action_to_take=0, angle=90):

        #Hidden
        self.max_speed = max_speed
        self.cal = init_cal
        self.damage = damage

        #Pysical feature that can be observed from outer existence
        self.hp = init_hp
        self.init_hp = init_hp
        self.type = type
        self.abs_angle = angle

        self.action = init_action_to_take

        #Observation that this creature observed
        self.observed_state = None

        #World-absolute feature
        self.xCoord = x
        self.yCoord = y
        self.state_info = turtle.Turtle()
        self.agent_visual = turtle.Turtle()

    def agent_policy(self):
        return

    def show_info(self):
        type_str = "Human"

        self.agent_visual.turtlesize(0.5, 1)

        self.state_info.penup()
        self.state_info.speed(0)
        self.state_info.color("white")
        self.state_info.setposition(w * self.xCoord-width/2, w * self.yCoord-width/2)
        infoString = "type: {0}, hp: {1}, quality: {2}".format(type_str, self.hp, self.cal)
        self.state_info.write(infoString, font=("Arial", 10, "normal"))

        self.agent_visual.color("yellow")
        self.agent_visual.shape("triangle")
        self.agent_visual.penup()
        self.agent_visual.speed(0)
        self.agent_visual.setposition(w*self.xCoord-width/2, w*self.yCoord-width/2)
        self.agent_visual.setheading(self.abs_angle)

    def record_state(self):

        f = open("transition"+str(self.type)+".txt", 'a', encoding='utf-8')
        f.write(str(self.xCoord)+" "+str(self.yCoord)+'\n')
        f.close()
        return

    def clear_img(self):
        self.agent_visual.hideturtle()
        self.agent_visual.clear()
        self.state_info.hideturtle()
        self.state_info.clear()

    def __del__(self):
        self.clear_img()