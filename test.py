

import numpy as np
import math
import random
from operator import itemgetter


a =[[0, 1], [2, 3], [4, 5], [6, 2]]


t =random.sample(range(len(a)), 2)

v = list(itemgetter(*t)(a))

print(v)