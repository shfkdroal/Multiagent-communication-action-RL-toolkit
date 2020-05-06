import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import graphic
from Classes import *
from Creature_policies import *

def init_learner():
    global is_learner_set
    global policy_learner
    if not is_learner_set:
        AgentList[0].observe_state()
        policy_learner = agent_net(physical_observation_shape=AgentList[0].observed_state_type.shape,
                                   should_load=should_load_, length_seq_w=AgentList[0].time_frame_w)
        is_learner_set = True
        return policy_learner

def xy_rand_generator(CoordList):
    x = random.randrange(0, coords_width - 1, 1)
    y = random.randrange(0, coords_width - 1, 1)
    coord = (int(x), int(y))
    if coord not in CoordList:
        CoordList.append(coord)
        return coord
    else:
        coord = xy_rand_generator(CoordList)
        return coord

def xy_rand_generator_obst(CoordList_obst):
    x = random.randrange(1, coords_width - 2, 1)
    y = random.randrange(1, coords_width - 2, 1)
    coord = (int(x), int(y))
    if coord not in CoordList_obst:
        CoordList_obst.append(coord)
        return coord
    else:
        coord = xy_rand_generator(CoordList_obst)
        return coord


human = None


def map_initializer(abs_coords, Obst_likelyhood, TypeList_obst, CoordList_obst, CoordList, CoordList_obst_detail):


    borderpen = turtle.Turtle()
    for i in range(int(coords_width)):
        for j in range(int(coords_width)):
            abs_coords[i, j] = Map_info(i, j, 0) #void



    #Obst type 1_1 and 1_2
    borderpen.color("gray")
    borderpen.speed(0)
    for i in range(random.randint(0, Obst_likelyhood)):
        obst_coord = xy_rand_generator_obst(CoordList_obst)
        obstacle_type = random.randint(0, 1)
        if obstacle_type == 0:
            obstacle_type = obstacle_type1_1
            TypeList_obst.append(0)
        else:
            obstacle_type = obstacle_type1_2
            TypeList_obst.append(1)

        for j in range(obstacle_type[0] + 1):
            borderpen.penup()
            newX = obst_coord[0] + j
            if newX >= coords_width:
                newX = coords_width - 1
            abs_coords[int(newX), int(obst_coord[1])].type = -2
            coord = (int(newX), int(obst_coord[1]))
            CoordList.append(coord)
            CoordList_obst_detail.append(coord)
            if j < obstacle_type[0]:
                borderpen.setposition(w * newX - width / 2, w * obst_coord[1] - width / 2)
                borderpen.setheading(0)
                borderpen.pendown()
                borderpen.fd(w)

    #Obst type 2_1
    for i in range(random.randint(0, Obst_likelyhood)):
        obst_coord = xy_rand_generator_obst(CoordList_obst)
        obstacle_type = random.randint(0, 1)
        if obstacle_type == 0:
            obstacle_type = obstacle_type2_1
            TypeList_obst.append(2)
        else:
            obstacle_type = obstacle_type2_2
            TypeList_obst.append(3)

        for j in range(obstacle_type[1] + 1):
            borderpen.penup()
            newY = obst_coord[1] + j
            if newY >= coords_width:
                newY = coords_width - 1
            abs_coords[int(obst_coord[0]), int(newY)].type = -2
            coord = (int(obst_coord[0]), int(newY))
            CoordList.append(coord)
            CoordList_obst_detail.append(coord)
            if j < obstacle_type[1]:
                borderpen.setposition(w * obst_coord[0] - width / 2, w * newY - width / 2)
                borderpen.setheading(90)
                borderpen.pendown()
                borderpen.fd(w)

    borderpen.penup()
    borderpen.hideturtle()


    if len(TypeList_obst) != len(CoordList_obst):
        TypeList_obst.clear()
        CoordList_obst.clear()
        CoordList = []
        CoordList_obst = []
        TypeList_obst = []
        map_initializer(abs_coords, Obst_likelyhood, TypeList_obst, CoordList_obst, CoordList)

    return borderpen





def spread_product(abs_coords, ExistenceList, CreatureList, AgentList, CoordList_obst_detail):
    global human

    global Num_Mammoth
    global Num_Tiger
    global Num_Insects
    global Num_Monkey
    global Num_Fruits
    global Num_P_Fruits
    global Num_Mouse
    global Num_Deer

    Num_Mammoth = 1  # 1
    Num_Tiger = 1#np.random.randint(0, 2)  # 2
    Num_Insects = 2  # 2
    Num_Monkey = 3 # 2
    Num_Fruits = 10  # 13
    Num_P_Fruits = 3  # 5
    Num_Mouse = 2  # 2
    Num_Deer = 2  # 2

    angles_ = [90, 180, -90, 0]
    print("CoordList_obst_detail: ", CoordList_obst_detail)
    CoordList = CoordList_obst_detail
    print("CoordList: ", CoordList)
    for i in range(Num_Deer):
        put_thing('deer', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Fruits):
        put_thing('fruit', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    print("CoordList: ", CoordList)
    for i in range(Num_Insects):
        put_thing('insects', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Mammoth):
        put_thing('mammoth', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Monkey):
        put_thing('monkey', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Mouse):
        put_thing('mouse', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_P_Fruits):
        put_thing('p-fruit', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Tiger):
        put_thing('tiger', angles_, abs_coords, CreatureList, ExistenceList, CoordList)

    #15, 5
    for i in range(Num_Agents):
        coords = xy_rand_generator(CoordList)
        agent = Agent(x=coords[0], y=coords[1], init_hp=15, max_speed=2, damage=5, init_cal=10,
                      type=Creature_Name_to_typeId["agent" + str(i+1)],
                   init_action_to_take=-1, init_com_action_to_take=-1, angle=angles_[random.randint(0, 3)], mental_param1=0.5, mental_param2=np.e)
        abs_coords[coords[0], coords[1]] = agent
        AgentList.append(agent)
        ExistenceList.append(agent)


    if human_agent_on:
        coords = xy_rand_generator(CoordList)
        human = Human(x=coords[0], y=coords[1], init_hp=15, max_speed=2, damage=5, init_cal=10, type=12,
                       init_action_to_take=0, angle=angles_[random.randint(0, 3)])
        abs_coords[coords[0], coords[1]] = human
        AgentList.append(human)
        ExistenceList.append(human)




def is_collide(modified_x, modified_y, one_step_modified_x, one_step_modified_y, abs_coords):

    if not (modified_x < coords_width and modified_y < coords_width \
            and modified_x >= 0 and modified_y >= 0):
        return True
    elif (not abs_coords[modified_x, modified_y].type == 0):
        return True
    elif abs_coords[one_step_modified_x, one_step_modified_y].type == -2 or \
            (abs_coords[modified_x, modified_y].type == -2):
        return True

    return False


def action_selection(t):

    for creature in CreatureList:
        creature.creature_policy()
    for agent in AgentList:
        agent.agent_policy(t, policy_learner)

def execute_all_the_actions_selected(ExistenceList, CreatureList, AgentList, abs_coords):

    np.random.shuffle(ExistenceList)
    for e in ExistenceList:
        execute_single_action(e, ExistenceList, CreatureList, AgentList, abs_coords)
    for agent in AgentList:
        agent.train(False, policy_learner)


def execute_single_action(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.prev_hp = copy.deepcopy(thing.hp)
    action = thing.action
    if action == 7:
        action = np.random.randint(0, 7)

    if action == 0:
        turn_left(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 1:
        turn_right(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 2:
        turn_up(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 3:
        turn_down(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 4:
        go_forward_normal(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 5:
        go_forward_fast(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    elif action == 6:
        if thing.type != Creature_Name_to_typeId['insects']:
            attack(thing, ExistenceList, CreatureList, AgentList, abs_coords)
        else:
            absorb(thing, ExistenceList, CreatureList, AgentList, abs_coords)


    if thing.type == Human_type:
        init_do_nothing()

    if (thing.type == Creature_Name_to_typeId['agent1'] or thing.type == Creature_Name_to_typeId['agent2'] or
            thing.type == Creature_Name_to_typeId['agent3']):
        thing.reset_interaction_arrays()

        # if (thing.type == Creature_Name_to_typeId['agent1']):
        #     print("agent{} action: {}, com: {}, word: {}".format(thing.type, action, thing.com_action, thing.said_word))
        # if (thing.type == Creature_Name_to_typeId['agent2']):
        #     print("agent{} action: {}, com: {}, word: {}".format(thing.type, action, thing.com_action, thing.said_word))

        global Whether_to_use_language
        if Whether_to_use_language:
            com_action = thing.com_action
            if com_action == 7:
                talk_to_agent1(thing)
            elif com_action == 8:
                talk_to_agent2(thing)
            elif com_action == 9:
                talk_to_agent3(thing)

            elif com_action == 10:
                express_emotion_to_agent1(thing)
            elif com_action == 11:
                express_emotion_to_agent2(thing)
            elif com_action == 12:
                express_emotion_to_agent3(thing)

        # if (thing.type == Creature_Name_to_typeId['agent1']) :
        #     print("wtw: {}".format(thing.who_said_what_to_who))
        # if (thing.type == Creature_Name_to_typeId['agent2']) :
        #     print("wtw: {}".format(thing.who_said_what_to_who))



def show_all(ExistenceList):

    for i, e in enumerate(ExistenceList):
        e.state_info.clear()
        e.show_info()

    wn.delay(0.2)


def put_thing(name, angles_, abs_coords, CreatureList, ExistenceList, CoordList):
    k = Creature_Name_to_typeId[name]
    coords = xy_rand_generator(CoordList)
    creature = Creatures(x=coords[0], y=coords[1], init_hp=init_hp[k - 1], max_speed=max_speed[k - 1],
                         damage=damage[k - 1],
                         init_cal=init_cal[k - 1], type=k,
                         init_action_to_take=-1, angle=angles_[random.randint(0, 3)])
    abs_coords[coords[0], coords[1]] = creature
    CreatureList.append(creature)
    ExistenceList.append(creature)

def eco_system_check_refill(abs_coords, CreatureList, ExistenceList, CoordList_obst_detail):

    global Num_Mammoth
    global Num_Tiger
    global Num_Insects
    global Num_Monkey
    global Num_Fruits
    global Num_P_Fruits
    global Num_Mouse
    global Num_Deer

    Num_Mammoth_current = 0
    Num_Tiger_current = 0
    Num_Insects_current = 0
    Num_Monkey_current = 0
    Num_Fruits_current = 0
    Num_P_Fruits_current = 0
    Num_Mouse_current = 0
    Num_Deer_current = 0

    for c in CreatureList:
        if c.type == Creature_Name_to_typeId['mammoth']:
            Num_Mammoth_current += 1
        elif c.type == Creature_Name_to_typeId['tiger']:
            Num_Tiger_current += 1
        elif c.type == Creature_Name_to_typeId['deer']:
            Num_Deer_current += 1
        elif c.type == Creature_Name_to_typeId['mouse']:
            Num_Mouse_current += 1
        elif c.type == Creature_Name_to_typeId['insects']:
            Num_Insects_current += 1
        elif c.type == Creature_Name_to_typeId['monkey']:
            Num_Monkey_current += 1
        elif c.type == Creature_Name_to_typeId['fruit']:
            Num_Fruits_current += 1
        elif c.type == Creature_Name_to_typeId['p-fruit']:
            Num_P_Fruits_current += 1

    coords = []
    search_shape = abs_coords.shape
    for i in range(search_shape[0]):
        for j in range(search_shape[1]):
            if abs_coords[i, j].type > 0:
                coord = (i, j)
                coords.append(coord)

    CoordList = coords + CoordList_obst_detail

    for i in range(Num_Mammoth - Num_Mammoth_current):
        put_thing('mammoth', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Tiger - Num_Tiger_current):
        put_thing('tiger', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Deer - Num_Deer_current):
        put_thing('deer', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Mouse - Num_Mouse_current):
        put_thing('mouse', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Insects - Num_Insects_current):
        put_thing('insects', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Monkey - Num_Monkey_current):
        put_thing('monkey', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_Fruits - Num_Fruits_current):
        put_thing('fruit', angles_, abs_coords, CreatureList, ExistenceList, CoordList)
    for i in range(Num_P_Fruits - Num_P_Fruits_current):
        put_thing('p-fruit', angles_, abs_coords, CreatureList, ExistenceList, CoordList)



########### action definition ###########

def dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords):

    if thing.hp <= 0:
        x = thing.xCoord
        y = thing.yCoord
        thing.clear_img()
        ExistenceList.remove(thing)
        if thing.type <= 8:
            CreatureList.remove(thing)
        else:
            thing.hp -= 100
            thing.train(True, policy_learner)
            AgentList.remove(thing)
        del thing
        abs_coords[x, y] = Map_info(x, y, 0)


def turn_left(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/1000
    thing.abs_angle = 180
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)

def turn_right(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/1000
    thing.abs_angle = 0
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)

def turn_up(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/1000
    thing.abs_angle = 90
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)

def turn_down(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/1000
    thing.abs_angle = -90
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)

def go_forward_normal(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    rad = np.deg2rad(thing.abs_angle)
    y_delt = int(np.sin(rad))
    x_delt = int(np.cos(rad))

    modified_x = thing.xCoord + x_delt
    modified_y = thing.yCoord + y_delt
    if not is_collide(modified_x, modified_y, modified_x, modified_y, abs_coords):
        thing.hp -= thing.init_hp / 300
        abs_coords[thing.xCoord, thing.yCoord] = Map_info(thing.xCoord, thing.yCoord, 0)
        thing.yCoord = modified_y
        thing.xCoord = modified_x
        abs_coords[thing.xCoord, thing.yCoord] = thing
        dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    else:
        thing.hp -= thing.init_hp / 300
        return

def go_forward_fast(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    if thing.hp < thing.init_hp/3:
        go_forward_normal(thing, ExistenceList, CreatureList, AgentList, abs_coords)
        return

    rad = np.deg2rad(thing.abs_angle)

    y_delt = int(np.sin(rad))
    x_delt = int(np.cos(rad))

    y_delt_ = thing.max_speed*y_delt
    x_delt_ = thing.max_speed*x_delt

    modified_x = thing.xCoord + x_delt_
    modified_y = thing.yCoord + y_delt_
    if not is_collide(modified_x, modified_y, thing.xCoord + x_delt, thing.yCoord + y_delt, abs_coords):
        thing.hp -= thing.init_hp / 150
        abs_coords[thing.xCoord, thing.yCoord] = Map_info(thing.xCoord, thing.yCoord, 0)
        thing.yCoord = modified_y
        thing.xCoord = modified_x
        abs_coords[thing.xCoord, thing.yCoord] = thing
        dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)
    else:
        thing.hp -= thing.init_hp / 150
        return

def attack(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/100
    rad = np.deg2rad(thing.abs_angle)
    y_delt = int(np.sin(rad))
    x_delt = int(np.cos(rad))

    #frontal coords
    target_x = thing.xCoord + x_delt
    target_y = thing.yCoord + y_delt

    if not (target_x < coords_width and target_y < coords_width
            and target_x >= 0 and target_y >= 0):
        dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)
        return

    target = abs_coords[target_x, target_y]
    if not target.type == 0:
        target.hp -= thing.damage
        if target.hp <= 0:
            thing.hp += target.cal
            target.clear_img()
            ExistenceList.remove(target)
            if target.type <= 8:
                CreatureList.remove(target)
            else:
                target.hp -= 100
                target.train(True, policy_learner)
                AgentList.remove(target)
            del target
            abs_coords[target_x, target_y] = Map_info(target_x, target_y, 0)
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)

def absorb(thing, ExistenceList, CreatureList, AgentList, abs_coords):
    thing.hp -= thing.init_hp/100
    rad = np.deg2rad(thing.abs_angle)
    y_delt = int(np.sin(rad))
    x_delt = int(np.cos(rad))

    #frontal coords
    target_x = thing.xCoord + x_delt
    target_y = thing.yCoord + y_delt

    if not (target_x < coords_width and target_y < coords_width
            and target_x >= 0 and target_y >= 0):
        dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)
        return

    target = abs_coords[target_x, target_y]
    if not target.type == 0:
        target.hp -= thing.damage
        target.cal -= thing.damage
        thing.hp += thing.damage
        if target.hp <= 0:
            thing.hp += target.cal
            target.clear_img()
            ExistenceList.remove(target)
            if target.type <= 8:
                CreatureList.remove(target)
            else:
                target.hp -= 100
                target.train(True, policy_learner)
                AgentList.remove(target)
            del target
            abs_coords[target_x, target_y] = Map_info(target_x, target_y, 0)
    dead_check(thing, ExistenceList, CreatureList, AgentList, abs_coords)


def talk_to_agent1(thing):

    speaker = thing.type
    word_spoken = thing.said_word

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = 0

    for a in AgentList:
        a.who_said_what_to_who[speaker_idx, target_idx] = word_spoken

def talk_to_agent2(thing):

    speaker = thing.type
    word_spoken = thing.said_word

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = Creature_Name_to_typeId['agent2'] - Creature_Name_to_typeId['agent1']

    for a in AgentList:
        a.who_said_what_to_who[speaker_idx, target_idx] = word_spoken

def talk_to_agent3(thing):

    speaker = thing.type
    word_spoken = thing.said_word

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = Creature_Name_to_typeId['agent3'] - Creature_Name_to_typeId['agent1']

    for a in AgentList:
        a.who_said_what_to_who[speaker_idx, target_idx] = word_spoken


def express_emotion_to_agent1(thing):
    speaker = thing.type
    emotion_expressed = thing.expressed_emotion
    emotion_intensity = thing.emotion_intensity

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = 0

    for a in AgentList:
        a.who_expressed_what_to_who[speaker_idx, target_idx] = emotion_expressed
        a.who_expressed_how_to_who[speaker_idx, target_idx] = emotion_intensity


def express_emotion_to_agent2(thing):
    speaker = thing.type
    emotion_expressed = thing.expressed_emotion
    emotion_intensity = thing.emotion_intensity

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = Creature_Name_to_typeId['agent2'] - Creature_Name_to_typeId['agent1']

    for a in AgentList:
        a.who_expressed_what_to_who[speaker_idx, target_idx] = emotion_expressed
        a.who_expressed_how_to_who[speaker_idx, target_idx] = emotion_intensity

def express_emotion_to_agent3(thing):
    speaker = thing.type
    emotion_expressed = thing.expressed_emotion
    emotion_intensity = thing.emotion_intensity

    speaker_idx = speaker - Creature_Name_to_typeId['agent1']
    target_idx = Creature_Name_to_typeId['agent3'] - Creature_Name_to_typeId['agent1']

    for a in AgentList:
        a.who_expressed_what_to_who[speaker_idx, target_idx] = emotion_expressed
        a.who_expressed_how_to_who[speaker_idx, target_idx] = emotion_intensity


########### action definition end ###########,

########### human action definition ###########


def turn_left_h():
    thing = human
    thing.action = 0


def turn_right_h():
    thing = human
    thing.action = 1


def turn_up_h():
    thing = human
    thing.action = 2


def turn_down_h():
    thing = human
    thing.action = 3


def go_forward_normal_h():
    thing = human
    thing.action = 4

def go_forward_fast_h():
    thing = human
    thing.action = 5

def attack_h():
    thing = human
    thing.action = 6

def init_do_nothing():
    thing = human
    thing.action = -1


########### human action definition end ###########

