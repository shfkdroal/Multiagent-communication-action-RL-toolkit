
import numpy as np
from Game_Param import *
import random


def is_collide_obstacle(modified_x, modified_y):
    if (not (modified_x < coords_width and modified_y < coords_width \
            and modified_x >= 0 and modified_y >= 0)) or (abs_coords[modified_x, modified_y].type < 0):
        return True
    return False

def is_collide_thing(modified_x, modified_y):
    if (not abs_coords[modified_x, modified_y].type == 0) or (abs_coords[modified_x, modified_y].type < 0):
        return True

    return False

def is_predator(self, obj):
    if self.type == Creature_Name_to_typeId['insects']:
        if obj.type == Creature_Name_to_typeId['monkey'] or obj.type == Creature_Name_to_typeId['mouse'] \
                or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['mouse']:
        if obj.type == Creature_Name_to_typeId['monkey'] or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['deer']:
        if obj.type == Creature_Name_to_typeId['tiger'] or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['monkey']:
        if obj.type == Creature_Name_to_typeId['tiger'] or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['mammoth']:
        if obj.type == Creature_Name_to_typeId['tiger'] or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['tiger']:
        if obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True

def is_target(self, obj):
    poison_confused = False
    coin = random.randint(0, 10)

    if coin < 1:
        poison_confused = obj.type == Creature_Name_to_typeId['p-fruit']

    if self.type == Creature_Name_to_typeId['insects']:
        if obj.type == Creature_Name_to_typeId['fruit'] or poison_confused:
            return True
    elif self.type == Creature_Name_to_typeId['mouse']:
        if obj.type == Creature_Name_to_typeId['insects']:
            return True
    elif self.type == Creature_Name_to_typeId['deer']:
        if obj.type == Creature_Name_to_typeId['fruit']  or poison_confused:
            return True
    elif self.type == Creature_Name_to_typeId['monkey']:
        if obj.type == Creature_Name_to_typeId['fruit'] or obj.type == Creature_Name_to_typeId['mouse']\
                or obj.type == Creature_Name_to_typeId['insects']  or poison_confused:
            return True
    elif self.type == Creature_Name_to_typeId['mammoth']:
        if obj.type == Creature_Name_to_typeId['fruit']  or poison_confused:
            return True
    elif self.type == Creature_Name_to_typeId['tiger']:
        if obj.type == Creature_Name_to_typeId['deer']\
                or obj.type == Creature_Name_to_typeId['monkey'] or \
                obj.type == Creature_Name_to_typeId['agent1'] or obj.type == Creature_Name_to_typeId['agent2'] or \
                obj.type == Creature_Name_to_typeId['agent3']:
            return True

def is_considered(self, obj):
    if self.type == Creature_Name_to_typeId['insects']:
        return False
    elif self.type == Creature_Name_to_typeId['mouse']:
        return False
    elif self.type == Creature_Name_to_typeId['deer']:
        return False

    elif self.type == Creature_Name_to_typeId['monkey']:
        if obj.type == Creature_Name_to_typeId['tiger'] or obj.type == Creature_Name_to_typeId['agent1'] or \
                obj.type == Creature_Name_to_typeId['agent2'] or obj.type == Creature_Name_to_typeId['agent3']:
            return True
    elif self.type == Creature_Name_to_typeId['mammoth']:
        if obj.type == Creature_Name_to_typeId['tiger']:
            return True
    elif self.type == Creature_Name_to_typeId['tiger']:
        if obj.type == Creature_Name_to_typeId['mammoth']:
            return True


def insect_policy(observed_state, self):

    half_range = self.half_range
    min_search_range = int(half_range / 4)
    running_th = min_search_range - 1
    mem = []
    for i in range(1, half_range + 1):
        for j in range(half_range - i, half_range + i):
            y = half_range - i
            point = observed_state[j, y]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i, half_range + i):
            x = half_range + i
            point = observed_state[x, j]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i + 1, half_range + i + 1):
            y = half_range + i
            point = observed_state[j, y]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i + 1, half_range + i + 1):
            x = half_range - i
            point = observed_state[x, j]
            if point.type > 0:
                mem.append(point)
        if mem and i >= min_search_range:
            break

    if not mem:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        min = np.abs(self.xCoord - mem[0].xCoord) + np.abs(self.yCoord - mem[0].yCoord)
        min_obj = mem[0]
        there_is_predator = False
        for i in range(1, len(mem)):
            is_predator_ = is_predator(self, mem[i])
            manhattan_dis = np.abs(self.xCoord - mem[i].xCoord) + np.abs(self.yCoord - mem[i].yCoord)
            if is_predator_:
                if not there_is_predator:
                    there_is_predator = True
                if manhattan_dis < min + running_th:
                    min = manhattan_dis - running_th
                    min_obj = mem[i]
            elif manhattan_dis < min:
                min = manhattan_dis
                min_obj = mem[i]

        if there_is_predator:
            speed = self.max_speed
        else:
            speed = 1
        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single
        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord and is_target(self, min_obj):
            return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width*3
        max_dist = 0
        min_action = -1
        max_action = -1

        two_step_collide = False
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)
                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if is_target(self, min_obj):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide:
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)







def deer_policy(observed_state, self):

    half_range = self.half_range
    min_search_range = int(half_range / 4)
    running_th = min_search_range - 1
    mem = []
    for i in range(1, half_range + 1):
        for j in range(half_range - i, half_range + i):
            y = half_range - i
            point = observed_state[j, y]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i, half_range + i):
            x = half_range + i
            point = observed_state[x, j]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i + 1, half_range + i + 1):
            y = half_range + i
            point = observed_state[j, y]
            if point.type > 0:
                mem.append(point)
        for j in range(half_range - i + 1, half_range + i + 1):
            x = half_range - i
            point = observed_state[x, j]
            if point.type > 0:
                mem.append(point)
        if mem and i >= min_search_range:
            break

    if not mem:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        min = np.abs(self.xCoord - mem[0].xCoord) + np.abs(self.yCoord - mem[0].yCoord)
        min_obj = mem[0]
        there_is_predator = False
        for i in range(1, len(mem)):
            is_predator_ = is_predator(self, mem[i])
            manhattan_dis = np.abs(self.xCoord - mem[i].xCoord) + np.abs(self.yCoord - mem[i].yCoord)
            if is_predator_:
                if not there_is_predator:
                    there_is_predator = True
                if manhattan_dis < min + running_th:
                    min = manhattan_dis - running_th
                    min_obj = mem[i]
            elif manhattan_dis < min:
                min = manhattan_dis
                min_obj = mem[i]

        if there_is_predator:
            speed = self.max_speed
        else:
            speed = 1
        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single
        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord and is_target(self, min_obj):
            return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width*3
        max_dist = 0
        min_action = -1
        max_action = -1

        two_step_collide = False
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)
                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if is_target(self, min_obj):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide:
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)




def mammoth_policy(observed_state, self):
    shape = observed_state.shape
    observed_state = np.reshape(observed_state, (1, shape[0] * shape[1]))[0]

    running_th = self.x_range_back_0

    min = coords_width * 3
    min_obj = None
    there_is_predator = False
    there_is_target = False
    not_zero_num = 0

    pred_hp_sum = 0
    pred_damage_sum = 0

    fellas_hp_sum = 0
    fellas_damage_sum = 0

    for i in range(observed_state.size):
        if observed_state[i].type > 0:
            if observed_state[i] != self:
                not_zero_num += 1
                is_predator_ = is_predator(self, observed_state[i])
                manhattan_dis = np.abs(self.xCoord - observed_state[i].xCoord) + np.abs(
                    self.yCoord - observed_state[i].yCoord)
                if is_predator_:
                    pred_hp_sum += observed_state[i].hp
                    pred_damage_sum += observed_state[i].damage
                    if not there_is_predator:
                        there_is_predator = True
                    if manhattan_dis < min + running_th:
                        min = manhattan_dis - running_th
                        min_obj = observed_state[i]
                elif manhattan_dis < min:
                    min = manhattan_dis
                    min_obj = observed_state[i]

                if (not there_is_target) and is_target(self, observed_state[i]):
                    there_is_target = True
            else:
                if observed_state[i].type == Creature_Name_to_typeId['mammoth']:
                    fellas_damage_sum += observed_state[i].damage
                    fellas_hp_sum += observed_state[i].hp

    if not_zero_num == 0:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        # print("detect type {}, pos: ({}, {})".format(min_obj.type, min_obj.xCoord, min_obj.yCoord))
        if there_is_predator or there_is_target:
            speed = self.max_speed
        else:
            speed = 1

        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single

        superior = (fellas_hp_sum * fellas_damage_sum > pred_damage_sum * pred_hp_sum)
        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord:
            if is_target(self, min_obj) or (superior and is_considered(self, min_obj)):
                return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width * 3
        max_dist = 0
        min_action = -1
        max_action = -1
        two_step_collide = False

        # angle selection
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)
                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if ((is_target(self, min_obj) or (superior and is_considered(self, min_obj))) and min_obj.type != self.type):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide:
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)



def tiger_policy(observed_state, self):
    shape = observed_state.shape
    observed_state = np.reshape(observed_state, (1, shape[0] * shape[1]))[0]

    running_th = self.x_range_back_0

    min = coords_width * 3
    min_obj = None
    there_is_predator = False
    there_is_target = False
    not_zero_num = 0

    pred_hp_sum = 0
    pred_damage_sum = 0

    fellas_hp_sum = 0
    fellas_damage_sum = 0

    for i in range(observed_state.size):
        if observed_state[i].type > 0:
            if observed_state[i] != self:
                not_zero_num += 1
                is_predator_ = is_predator(self, observed_state[i])
                manhattan_dis = np.abs(self.xCoord - observed_state[i].xCoord) + np.abs(
                    self.yCoord - observed_state[i].yCoord)
                if is_predator_:
                    pred_hp_sum += observed_state[i].hp
                    pred_damage_sum += observed_state[i].damage
                    if not there_is_predator:
                        there_is_predator = True
                    if manhattan_dis < min + running_th:
                        min = manhattan_dis - running_th
                        min_obj = observed_state[i]
                elif manhattan_dis < min:
                    min = manhattan_dis
                    min_obj = observed_state[i]

                if (not there_is_target) and is_target(self, observed_state[i]):
                    there_is_target = True
            else:
                if observed_state[i].type == Creature_Name_to_typeId['tiger']:
                    fellas_damage_sum += observed_state[i].damage
                    fellas_hp_sum += observed_state[i].hp

    if not_zero_num == 0:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        # print("detect type {}, pos: ({}, {})".format(min_obj.type, min_obj.xCoord, min_obj.yCoord))
        if there_is_predator or there_is_target:
            speed = self.max_speed
        else:
            speed = 1

        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single
        superior = (fellas_hp_sum * fellas_damage_sum > pred_damage_sum * pred_hp_sum)

        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord:
            if is_target(self, min_obj) or (superior and is_considered(self, min_obj)):
                return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width * 3
        max_dist = 0
        min_action = -1
        max_action = -1
        two_step_collide = False
        min_dist_frontal = min_dist

        # angle selection
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)

                modified_x_frontal = self.xCoord + x_delt_single
                modified_y_frontal = self.yCoord + y_delt_single
                manhattan_dis_frontal = np.abs(min_obj.xCoord - modified_x_frontal) + np.abs(min_obj.yCoord - modified_y_frontal)
                if min_dist_frontal >= manhattan_dis_frontal:
                    min_dist_frontal = manhattan_dis_frontal
                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if ((is_target(self, min_obj) or (superior and is_considered(self, min_obj))) and min_obj.type != self.type):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide or (min_dist_frontal <= min_dist):
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)





def monkey_policy(observed_state, self):
    shape = observed_state.shape
    observed_state = np.reshape(observed_state, (1, shape[0] * shape[1]))[0]

    running_th = self.x_range_back_0

    min = coords_width * 3
    min_obj = None
    there_is_predator = False
    there_is_target = False
    not_zero_num = 0

    pred_hp_sum = 0
    pred_damage_sum = 0

    fellas_hp_sum = 0
    fellas_damage_sum = 0

    for i in range(observed_state.size):
        if observed_state[i].type > 0:
            if observed_state[i] != self:
                not_zero_num += 1
                is_predator_ = is_predator(self, observed_state[i])
                manhattan_dis = np.abs(self.xCoord - observed_state[i].xCoord) + np.abs(
                    self.yCoord - observed_state[i].yCoord)
                if is_predator_:
                    pred_hp_sum += observed_state[i].hp
                    pred_damage_sum += observed_state[i].damage
                    if not there_is_predator:
                        there_is_predator = True
                    if manhattan_dis < min + running_th:
                        min = manhattan_dis - running_th
                        min_obj = observed_state[i]
                elif manhattan_dis < min:
                    min = manhattan_dis
                    min_obj = observed_state[i]

                if (not there_is_target) and is_target(self, observed_state[i]):
                    there_is_target = True
            else:
                if observed_state[i].type == Creature_Name_to_typeId['monkey']:
                    fellas_damage_sum += observed_state[i].damage
                    fellas_hp_sum += observed_state[i].hp

    if not_zero_num == 0:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        # print("detect type {}, pos: ({}, {})".format(min_obj.type, min_obj.xCoord, min_obj.yCoord))
        if there_is_predator or there_is_target:
            speed = self.max_speed
        else:
            speed = 1

        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single

        superior = (fellas_hp_sum * fellas_damage_sum > pred_damage_sum * pred_hp_sum)
        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord:
            if is_target(self, min_obj) or (superior and is_considered(self, min_obj)):
                return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width * 3
        max_dist = 0
        min_action = -1
        max_action = -1
        two_step_collide = False
        min_dist_frontal = min_dist

        # angle selection
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)

                modified_x_frontal = self.xCoord + x_delt_single
                modified_y_frontal = self.yCoord + y_delt_single
                manhattan_dis_frontal = np.abs(min_obj.xCoord - modified_x_frontal) + np.abs(min_obj.yCoord - modified_y_frontal)

                if min_dist_frontal >= manhattan_dis_frontal:
                    min_dist_frontal = manhattan_dis_frontal
                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if ((is_target(self, min_obj) or (superior and is_considered(self, min_obj))) and min_obj.type != self.type):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide or (min_dist_frontal <= min_dist):
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)




def mouse_policy(observed_state, self):

    shape = observed_state.shape
    observed_state = np.reshape(observed_state, (1, shape[0] * shape[1]))[0]

    running_th = self.x_range_back_0

    min = coords_width*3
    min_obj = None
    there_is_predator = False
    there_is_target = False
    not_zero_num = 0
    for i in range(observed_state.size):
        if observed_state[i].type > 0 and observed_state[i] != self:
            not_zero_num += 1
            is_predator_ = is_predator(self, observed_state[i])
            manhattan_dis = np.abs(self.xCoord - observed_state[i].xCoord) + np.abs(
                self.yCoord - observed_state[i].yCoord)
            if is_predator_:
                if not there_is_predator:
                    there_is_predator = True
                if manhattan_dis < min + running_th:
                    min = manhattan_dis - running_th
                    min_obj = observed_state[i]
            elif manhattan_dis < min:
                min = manhattan_dis
                min_obj = observed_state[i]

            if (not there_is_target) and is_target(self, observed_state[i]):
                there_is_target = True

    if not_zero_num == 0:
        coint = random.randint(0, 10)
        if coint < 3:
            return random.randint(0, 3)
        else:
            return random.randint(4, 5)
    else:
        #print("detect type {}, pos: ({}, {})".format(min_obj.type, min_obj.xCoord, min_obj.yCoord))
        if there_is_predator or there_is_target:
            speed = self.max_speed
        else:
            speed = 1

        rad = np.deg2rad(self.abs_angle)
        y_delt_single = int(np.sin(rad))
        x_delt_single = int(np.cos(rad))
        modified_x_frontal = self.xCoord + x_delt_single
        modified_y_frontal = self.yCoord + y_delt_single
        if modified_x_frontal == min_obj.xCoord and modified_y_frontal == min_obj.yCoord and is_target(self, min_obj):
            return 6  # attack

        min_angle = 90
        max_angle = 90

        min_dist = coords_width*3
        max_dist = 0
        min_action = -1
        max_action = -1
        two_step_collide = False
        min_dist_frontal = min_dist

        # angle selection
        for i, a in enumerate(angles_):
            rad = np.deg2rad(a)

            y_delt_single = int(np.sin(rad))
            x_delt_single = int(np.cos(rad))

            y_delt = speed * y_delt_single
            x_delt = speed * x_delt_single

            modified_x = self.xCoord + x_delt
            modified_y = self.yCoord + y_delt

            if not is_collide_obstacle(modified_x, modified_y):
                manhattan_dis = np.abs(min_obj.xCoord - modified_x) + np.abs(min_obj.yCoord - modified_y)

                modified_x_frontal = self.xCoord + x_delt_single
                modified_y_frontal = self.yCoord + y_delt_single
                manhattan_dis_frontal = np.abs(min_obj.xCoord - modified_x_frontal) + np.abs(min_obj.yCoord - modified_y_frontal)
                if min_dist_frontal >= manhattan_dis_frontal:
                    min_dist_frontal = manhattan_dis_frontal

                if min_dist >= manhattan_dis:
                    min_dist = manhattan_dis
                    min_action = i
                    min_angle = a
                    two_step_collide = is_collide_thing(modified_x, modified_y)
                if max_dist <= manhattan_dis:
                    max_dist = manhattan_dis
                    max_action = i
                    max_angle = a

        if is_target(self, min_obj):
            if min_angle == self.abs_angle:
                if speed == 1 or two_step_collide or (min_dist_frontal <= min_dist):
                    return 4  # move forward normal
                else:
                    return 5  # move forward fast

            else:
                return min_action

        elif there_is_predator:
            if max_angle == self.abs_angle:
                return 5  # move forward fast
            else:
                return max_action
        else:
            coint = random.randint(0, 10)
            if coint < 3:
                return random.randint(0, 3)
            else:
                return random.randint(4, 5)











