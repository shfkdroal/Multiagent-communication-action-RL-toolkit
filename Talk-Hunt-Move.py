
import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import graphic
from World_manager import *
from Classes import *
import csv

borderpen = turtle.Turtle()
borderpen.speed(0)
borderpen.color("white")
borderpen.penup()
borderpen.setposition(-width/2, -width/2)
borderpen.pendown()
borderpen.pensize(3)

for side in range(4):
    borderpen.fd(width)
    borderpen.lt(90)

borderpen.pensize(0.5)
WxCoord = -width/2 #-300
WyCoord = -width/2 #-300


"""
for i in range(int(width/w)):
    WxCoord = WxCoord + w
    borderpen.color("gray")
    borderpen.penup()
    borderpen.setposition(WxCoord, -width/2)
    borderpen.setheading(90)
    borderpen.pendown()
    borderpen.fd(width)
for i in range(int(width/w)):
    WyCoord = WyCoord + w
    borderpen.color("gray")
    borderpen.penup()
    borderpen.setposition(-width/2, WyCoord)
    borderpen.setheading(0)
    borderpen.pendown()
    borderpen.fd(width)
"""

borderpen.hideturtle()

borderpen_ep = turtle.Turtle()
borderpen_ep.speed(0)
borderpen_ep.color("white")
borderpen_ep.penup()
borderpen_ep.setposition(-width/2 -width/5, width/2 + width/5)
borderpen_ep.pendown()
borderpen_ep.pensize(3)
borderpen_ep.hideturtle()

###########################


Num_Episodes = 100

epsil_list = [100, 70, 50, 20, 0]

def start():
    global Epsilon
    global init_Epsil
    global abs_coords
    global AgentList
    global CreatureList
    global ExistenceList
    global CoordList
    global CoordList_obst
    global TypeList_obst
    global Obst_likelyhood
    global CoordList_obst_detail

    policy_learner_ = None
    is_learner_set_ = False

    for e in range(Num_Episodes):
        f1 = open('./reward' + str(Creature_Name_to_typeId['agent1']) + '.csv', 'a', newline='')
        makewrite1 = csv.writer(f1)
        f2 = open('./reward' + str(Creature_Name_to_typeId['agent2']) + '.csv', 'a', newline='')
        makewrite2 = csv.writer(f2)
        f3 = open('./reward' + str(Creature_Name_to_typeId['agent3']) + '.csv', 'a', newline='')
        makewrite3 = csv.writer(f3)

        print("init map start!")
        borderpen_ep.clear()
        borderpen_ep.write("Episode {}, Epsil {}".format(e, Epsilon))

        border = map_initializer(abs_coords, Obst_likelyhood, TypeList_obst, CoordList_obst,
                                 CoordList, CoordList_obst_detail)
        spread_product(abs_coords, ExistenceList, CreatureList, AgentList, CoordList_obst_detail)
        line = ''
        for i in range(int(coords_width)):
            for j in range(int(coords_width)):
                line += (' ' + (str(abs_coords[i, j].type)))
            print(line)
            line = ''

        print("init map done!")

        if not is_learner_set_:
            policy_learner_ = init_learner()
            is_learner_set_ = True

        total_reward_per_episode1 = 0
        total_reward_per_episode2 = 0
        total_reward_per_episode3 = 0

        total_lifeexpec_per_episode1 = 0
        total_lifeexpec_per_episode2 = 0
        total_lifeexpec_per_episode3 = 0
        for i in range(700):
            action_selection(i) #fillout the word states based on the array. Specify the target of the previous (previous state) to action
            execute_all_the_actions_selected(ExistenceList, CreatureList, AgentList, abs_coords)
            show_all(ExistenceList)

            if i % 50 == 0:
                eco_system_check_refill(abs_coords, CreatureList, ExistenceList, CoordList_obst_detail)
                for i in range(int(coords_width)):
                    for j in range(int(coords_width)):
                        line += (' ' + (str(abs_coords[i, j].type)))
                    print(line)
                    line = ''
                print("ecosystem maintained!")


            if len(AgentList) == 0:
                break
            for a in AgentList:
                if Creature_Name_to_typeId['agent1'] == a.type:
                    total_reward_per_episode1 += a.reward
                    total_lifeexpec_per_episode1 = i
                elif Creature_Name_to_typeId['agent2'] == a.type:
                    total_reward_per_episode2 += a.reward
                    total_lifeexpec_per_episode2 = i
                elif Creature_Name_to_typeId['agent3'] == a.type:
                    total_reward_per_episode3 += a.reward
                    total_lifeexpec_per_episode3 = i

        reset(border)
        makewrite1.writerow([e, total_reward_per_episode1, total_lifeexpec_per_episode1, total_lifeexpec_per_episode1*(total_reward_per_episode1 + 15)/10, Epsilon])
        f1.close()
        makewrite2.writerow([e, total_reward_per_episode2, total_lifeexpec_per_episode2, total_lifeexpec_per_episode2*(total_reward_per_episode2 + 15)/10, Epsilon])
        f2.close()
        makewrite3.writerow([e, total_reward_per_episode3, total_lifeexpec_per_episode3, total_lifeexpec_per_episode3*(total_reward_per_episode3 + 15)/10, Epsilon])
        f3.close()
        # Epsilon -= 1
        Epsilon = epsil_list[np.random.randint(0, 5)]
        if Epsilon < 0:
            Epsilon = init_Epsil

        if e % 20 == 0:
            policy_learner_.saver.save(policy_learner_.sess, './model/agent_policy_model' + str(e))
    policy_learner_.saver.save(policy_learner_.sess, './model/agent_policy_model_final')
    return

def start_without_agent():

    global abs_coords
    global AgentList
    global CreatureList
    global ExistenceList
    global CoordList
    global CoordList_obst
    global TypeList_obst
    global Obst_likelyhood
    global CoordList_obst_detail


    border = map_initializer(abs_coords, Obst_likelyhood, TypeList_obst, CoordList_obst, CoordList, CoordList_obst_detail)
    spread_product(abs_coords, ExistenceList, CreatureList, AgentList, CoordList_obst_detail)
    num_itr = 500

    line = ''
    for i in range(int(coords_width)):
        for j in range(int(coords_width)):
            line += (' '+ (str(abs_coords[i, j].type)))
        print(line)
        line = ''

    for i in range(num_itr):
        action_selection(
            i)  # fillout the word states based on the array. Specify the target of the previous (previous state) to action
        execute_all_the_actions_selected(ExistenceList, CreatureList, AgentList, abs_coords)
        show_all(ExistenceList)
        if i % 50 == 0:
            eco_system_check_refill(abs_coords, CreatureList, ExistenceList, CoordList_obst_detail)

    reset(border)
    del border




def reset(border):
    global CoordList
    global TypeList_obst
    global CoordList_obst
    global CoordList_obst_detail
    global CreatureList
    global AgentList
    global ExistenceList
    global abs_coords

    print("reset start!")
    for c in CreatureList:
        c.clear_img()
    for a in AgentList:
        a.clear_img()
    for e in ExistenceList:
        e.clear_img()
    CoordList.clear()
    TypeList_obst.clear()
    CoordList_obst.clear()
    CoordList_obst_detail.clear()
    CreatureList.clear()
    AgentList.clear()
    ExistenceList.clear()
    border.clear()

    abs_coords = np.zeros((int(coords_width), int(coords_width)), dtype=object)

    #print("AgentList: ", AgentList)
    print("reset complete!: ")


def shift_showing_mode1():
    global showing_mode
    showing_mode[1] = not showing_mode[1]
    return
def shift_showing_mode2():
    global showing_mode
    showing_mode[2] = not showing_mode[2]
    return
def shift_showing_mode3():
    global showing_mode
    showing_mode[3] = not showing_mode[3]
    return
def shift_showing_mode4():
    global showing_mode
    showing_mode[4] = not showing_mode[4]
    return
def shift_showing_mode5():
    global showing_mode
    showing_mode[5] = not showing_mode[5]
    return
def shift_showing_mode6():
    global showing_mode
    showing_mode[6] = not showing_mode[6]
    return
def shift_showing_mode7():
    global showing_mode
    showing_mode[7] = not showing_mode[7]
    return
def shift_showing_mode8():
    global showing_mode
    showing_mode[8] = not showing_mode[8]
    return
def shift_showing_mode9():
    global showing_mode
    showing_mode[9] = not showing_mode[9]
    return
def shift_showing_mode10():
    global showing_mode
    showing_mode[10] = not showing_mode[10]
    return
def shift_showing_mode11():
    global showing_mode
    showing_mode[11] = not showing_mode[11]
    return

#create keyboard bindings
if Num_Agents > 0:
    wn.onkeypress(start, 's')
else:
    wn.onkeypress(start_without_agent, 's')

wn.onkeypress(shift_showing_mode1, '1')
wn.onkeypress(shift_showing_mode2, '2')
wn.onkeypress(shift_showing_mode3, '3')
wn.onkeypress(shift_showing_mode4, '4')
wn.onkeypress(shift_showing_mode5, '5')
wn.onkeypress(shift_showing_mode6, '6')
wn.onkeypress(shift_showing_mode7, '7')
wn.onkeypress(shift_showing_mode8, '8')
wn.onkeypress(shift_showing_mode9, 'q')
wn.onkeypress(shift_showing_mode10, 'w')
wn.onkeypress(shift_showing_mode11, 'e')

if human_agent_on:
    wn.onkeyrelease(init_do_nothing, 'Left')
    wn.onkeyrelease(init_do_nothing, 'Right')
    wn.onkeyrelease(init_do_nothing, 'Up')
    wn.onkeyrelease(init_do_nothing, 'Down')
    wn.onkeyrelease(init_do_nothing, 'q')
    wn.onkeyrelease(init_do_nothing, 'w')
    wn.onkeyrelease(init_do_nothing, 'a')


wn.listen()
wn.mainloop()


