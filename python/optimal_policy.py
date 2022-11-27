"""
This file calls methods in mdp.py to calculate the optimal policy for a given gridworld
"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, MDP'
__email__ = 'Josh.Cu@gmail.com'

import time
import helpers
from mdp import MDP
import sys

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)
    start_time = time.time()
    m = MDP(filename=sys.argv[1])
    #m = MDP()

    utility = m.value_iteration(m.current_utility, True)
    p = m.get_optimal_policy(utility)
    helpers.print_grid(utility, m)
    #helpers.print_grid(p, m, True)
    print("Time taken: ", time.time() - start_time)
    helpers.print_grid_arrows(p, m)
    # helpers.write_policy(helpers.assignment_out(p))
