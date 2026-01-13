This repository contains the code which implements the algorithm designed in the paper Learning Algorithm for Mean Field Control by H. Mete Soner, Josef Teichmann and Qinxin Yan.

In Crowd2DMFC.py, it models a 2-dimensional mean field control problem, which solves an optimal transport problem while avoiding the crowd during the transportation. The dependece on the measure is through the inital and terminal distrbutions and the crowd distribution.

In Simulate_Particles.py, we simulate the dynamics of particles using the optimal control(the trained neural network) under independetly simulated noises.

In KuramotoMFC and solver.py, we implement the mean field control algorithm to Kuramoto mean field control problem. 
