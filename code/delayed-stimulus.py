# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from model import *

# --- Parameter
ms         = 0.001
settling   = 500*ms
trial      = 2500*ms
dt         = 1*ms

getWeights = True
getWeights = False
debug      = True
threshold  = 40
alpha_c    = 0.025
alpha_LTP  = 0.004
alpha_LTD  = 0.002
Wmin, Wmax = 0.25, 0.75
tau        = 0.01
clamp      = Clamp(min=0, max=1000)
sigmoid    = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)

CTX = AssociativeStructure(
                 tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
STR = AssociativeStructure(
                 tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
STN = Structure( tau=tau, rest=-10.0, noise=0.001, activation=clamp )
GPI = Structure( tau=tau, rest=+10.0, noise=0.030, activation=clamp )
THL = Structure( tau=tau, rest=-40.0, noise=0.001, activation=clamp )
structures = (CTX, STR, STN, GPI, THL)

def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)


# These weights will change (learning)
if getWeights:
    W = weights(4)
else:
    W = np.load("delayed-stimulus.npy")

connections = [
    OneToOne( CTX.cog.V, STR.cog.Isyn, W,            gain=+1.0 ),
    OneToOne( CTX.mot.V, STR.mot.Isyn, weights(4),   gain=+1.0 ),
    OneToOne( CTX.ass.V, STR.ass.Isyn, weights(4*4), gain=+1.0 ),
    CogToAss( CTX.cog.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    MotToAss( CTX.mot.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    OneToOne( CTX.cog.V, STN.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.mot.V, STN.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( STR.cog.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    OneToOne( STR.mot.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    AssToCog( STR.ass.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    AssToMot( STR.ass.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    OneToAll( STN.cog.V, GPI.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToAll( STN.mot.V, GPI.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( GPI.cog.V, THL.cog.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( GPI.mot.V, THL.mot.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( THL.cog.V, CTX.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( THL.mot.V, CTX.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.cog.V, THL.cog.Isyn, np.ones(4),   gain=+0.4 ),
    OneToOne( CTX.mot.V, THL.mot.Isyn, np.ones(4),   gain=+0.4 ),
]


cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

def set_trial():
    global cues_mot, cues_cog, cues_values, cues_reward

    np.random.shuffle(cues_cog)
    np.random.shuffle(cues_mot)
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]
    v = 7
    noise = 0.01
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0
    CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
    CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)

def first_stimulus():
    global cues_mot, cues_cog, cues_values, cues_reward
    global dc1
    np.random.shuffle(cues_cog)
    np.random.shuffle(cues_mot)
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]
    c1 = max(c1,c2) 
    dc1 = c1
    m1 = cues_mot[0]
    v = 7
    noise = 0.01
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0
    CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)

def second_stimulus():
    global dc2
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]
    c2 = min(c1,c2) 
    dc2 = c2
    m2 = cues_mot[1]
    v = 7
    noise = 0.01
    CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)

def stop_trial():
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)


def iterate(dt):
    global connections, structures

    # Flush connections
    for connection in connections:
        connection.flush()

    # Propagate activities
    for connection in connections:
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)


def reset():
    global cues_values, structures
    for structure in structures:
        structure.reset()


def learn(time, debug=True):
    # A motor decision has been made
    cc1, cc2 = cues_cog[:2]
    if getWeights:
        c1 = cc1
        c2 = cc2
    else:
        c1 = max(cc1,cc2)
        c2 = min(cc1,cc2)
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(CTX.mot.V)
    cog_choice = np.argmax(CTX.cog.V)

    # The actual cognitive choice may differ from the cognitive choice
    # Only the motor decision can designate the chosen cue
    if mot_choice == m1:
        choice = c1
    else:
        choice = c2

    if choice == min(c1,c2):
        P.append(1)
    else:
        P.append(0)

    # Compute reward
    reward = np.random.uniform(0,1) < cues_reward[choice]
    R.append(reward)

    # Compute prediction error
    #error = cues_reward[choice] - cues_value[choice]
    error = reward - cues_value[choice]

    # Update cues values
    cues_value[choice] += error* alpha_c

    # Learn
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * STR.cog.V[choice]
    W[choice] = W[choice] + dw * (W[choice]-Wmin)*(Wmax-W[choice])

    if not debug: return

    # Just for displaying ordered cue
    oc1,oc2 = min(c1,c2), max(c1,c2)
    if choice == oc1:
        print "Choice:          [%d] / %d  (good)" % (oc1,oc2)
    else:
        print "Choice:           %d / [%d] (bad)" % (oc1,oc2)
    print "Reward (%3d%%) :   %d" % (int(100*cues_reward[choice]),reward)
    print "Mean performance: %.3f" % np.array(P)[-20:].mean()
    print "Mean reward:      %.3f" % np.array(R).mean()
    print "Response time:    %d ms" % (time)

delay = 50
 

P, R = [], []
# 120 trials
for j in range(120):
    reset()

    # Settling phase (500ms)
    i0 = 0
    i1 = i0+int(settling/dt)
    for i in xrange(i0,i1):
        iterate(dt)

    # Trial setup
    if getWeights:
        set_trial()
    else:
        first_stimulus()
    # Learning phase (2500ms)
    i0 = int(settling/dt)
    i1 = i0+int(trial/dt)
    for i in xrange(i0,i1):
        if not getWeights:
            if i == 500 + delay:
                print "introducing second stimulus"
                second_stimulus()
        iterate(dt)
        # Test if a decision has been made
        if CTX.mot.delta > threshold:
            learn(time=i-500, debug=debug)
            break

    # Debug information
    if debug:
        if i >= (i1-1):
            print "! Failed trial"
        print

print "Done with learning. Running a trial to check decision time"
# -----------------------------------------------------------------------------
#from display import *
def display_ctx(history, cues, delay=0, duration=3.0, filename=None):
    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    timesteps = np.linspace(0,duration, len(history))

    fig.patch.set_facecolor('.9')
    ax = plt.subplot(1,1,1)
    d = "Delay "+str(delay)+"ms, Cues-[ "+str(cues)+" ]"
    plt.plot(timesteps, history["CTX"]["cog"][:,0],c='r', label="Cognitive Cortex "+d)
    plt.text(1.5, history["CTX"]["cog"][:,0][1500], "CUE-0", size=10, rotation=0.,
         ha="center", va="center",
         bbox = dict(ec='1',fc='1')
        )
    plt.plot(timesteps, history["CTX"]["cog"][:,1],c='r')
    plt.text(1.5, history["CTX"]["cog"][:,1][1500], "CUE-1", size=10, rotation=0.,
         ha="center", va="center",
         bbox = dict(ec='1',fc='1')
        )
    plt.plot(timesteps, history["CTX"]["cog"][:,2],c='r')
    plt.text(1.5, history["CTX"]["cog"][:,2][1500], "CUE-2", size=10, rotation=0.,
         ha="center", va="center",
         bbox = dict(ec='1',fc='1')
        )
    plt.plot(timesteps, history["CTX"]["cog"][:,3],c='r')
    plt.text(1.5, history["CTX"]["cog"][:,3][1500], "CUE-3", size=10, rotation=0.,
         ha="center", va="center",
         bbox = dict(ec='1',fc='1')
        )
    plt.plot(timesteps, history["CTX"]["mot"][:,0],c='b', label="Motor Cortex")
    plt.plot(timesteps, history["CTX"]["mot"][:,1],c='b')
    plt.plot(timesteps, history["CTX"]["mot"][:,2],c='b')
    plt.plot(timesteps, history["CTX"]["mot"][:,3],c='b')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0.0,duration)
    plt.ylim(0.0,60.0)

    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])

    if filename is not None:
        plt.savefig(filename)
    plt.show()

if getWeights:
    print "saving weights"
    np.save("delayed-stimulus.npy", W)
else:
    dt = 0.001
    for j in range(1):
        reset()
        for i in xrange(0,500):
            iterate(dt)
        first_stimulus()
        for i in xrange(500,2500):
            if i == 500 + delay:
                print "introducing second stimulus"
                second_stimulus()
            iterate(dt)
        stop_trial()
        for i in xrange(2500,3000):
            iterate(dt)

    dtype = [ ("CTX", [("mot", float, 4), ("cog", float, 4), ("ass", float, 16)]),
              ("STR", [("mot", float, 4), ("cog", float, 4), ("ass", float, 16)]),
              ("GPI", [("mot", float, 4), ("cog", float, 4)]),
              ("THL", [("mot", float, 4), ("cog", float, 4)]),
              ("STN", [("mot", float, 4), ("cog", float, 4)])]

    history = np.zeros(3000, dtype=dtype)
    history["CTX"]["mot"]   = CTX.mot.history[:3000]
    history["CTX"]["cog"]   = CTX.cog.history[:3000]
    history["CTX"]["ass"]   = CTX.ass.history[:3000]
    history["STR"]["mot"] = STR.mot.history[:3000]
    history["STR"]["cog"] = STR.cog.history[:3000]
    history["STR"]["ass"] = STR.ass.history[:3000]
    history["STN"]["mot"]      = STN.mot.history[:3000]
    history["STN"]["cog"]      = STN.cog.history[:3000]
    history["GPI"]["mot"]      = GPI.mot.history[:3000]
    history["GPI"]["cog"]      = GPI.cog.history[:3000]
    history["THL"]["mot"] = THL.mot.history[:3000]
    history["THL"]["cog"] = THL.cog.history[:3000]

    if 1:
        display_ctx(history, np.array([dc1, dc2]), delay, 3.0, "after-learning-delay-cortex.pdf")
    #if 1:
    #    display_all(history, 3.0, "after-learning-all-1bis.pdf")




