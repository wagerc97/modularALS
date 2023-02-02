"""
Source: https://pymoo.org/problems/single/zakharov.html

"""

import numpy as np

from pymoo.problems import get_problem
from pymoo.visualization.fitness_landscape import FitnessLandscape


problem = get_problem("zakharov", n_var=2)

FitnessLandscape(problem, angle=(45, 45), _type="surface").show()

