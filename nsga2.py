"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Example Usage from PyMOO documentation:
- We refer here to our documentation for all the details. However, for instance, executing NSGA2:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#### Common error with tutorial ####
# Error: Import error of get_problem
# Instead of from pymoo.problems import get_problem use from pymoo.problems.multi import * .

# And for get_problem use problem instead. As an example:
#> get_problem("zdt1").pareto_front()
# Should be converted to:
#> ZDT1().pareto_front()

from pymoo.algorithms.moo.nsga2 import NSGA2
#from pymoo.problems import get_problem
from pymoo.problems.multi import *
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#problem = get_problem("zdt1")
problem = ZDT1().pareto_front()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()