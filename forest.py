import mdptoolbox.example
import numpy as np

np.random.seed(0)

print "*********Value Iteration**********"

P,R = mdptoolbox.example.forest()
vi = mdptoolbox.mdp.ValueIteration(P,R,0.9)
vi.setVerbose()
vi.run()
print vi.policy

print "**********Policy Iteration********"

pi = mdptoolbox.mdp.PolicyIteration(P,R,0.9)
pi.setVerbose()
pi.run()
print pi.policy

print "********Q Learning ********"
ql = mdptoolbox.mdp.PolicyIteration(P,R,0.9)
ql.setVerbose()
ql.run()
print ql.policy
