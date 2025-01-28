from ccw import ccw
from steer import steer
from chk_collision import chk_collision
from dist import dist
from no_collision import no_collision
from plot_obstacle_poly import plot_obstacle_poly

import numpy as np
import matplotlib.pyplot as plt

# Set up environment
x_max = 100
y_max = 100
EPS = 1
numNodes = 500

poly = [
    np.array([[50, 15], [50, 35], [70, 35], [70, 15]]),
    np.array([[50, 45], [50, 65], [70, 65], [70, 45]]),
    np.array([[20, 45], [10, 30], [28, 32]]),
    np.array([[20, 60], [45, 60], [45, 53], [20, 53]])
]

q_start = {'coord': [5, 60], 'cost': 0, 'parent': 0}
q_goal = {'coord': [80, 30], 'cost': 0}

nodes = [q_start]

fig, ax = plt.subplots()
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)

plot_obstacle_poly(ax, poly, "black")
ax.plot(80, 30, 'go', markersize=10, markerfacecolor='g')
ax.plot(5, 60, 'ro', markersize=10, markerfacecolor='r')

plt.ion()
plt.show()

result1 = []
elapsed_time1 = []

for i in range(numNodes):
    K1 = np.random.rand()
    thenorm1 = np.linalg.norm(K1)
    
    q_rand = [np.random.randint(x_max), np.random.randint(y_max)]
    ax.plot(q_rand[0], q_rand[1], 'x', color=[0, 0.4470, 0.7410])
    plt.pause(0.01)
    
    if any(np.array_equal(node['coord'], q_goal['coord']) for node in nodes):
        break
    
    ndist = [dist(node['coord'], q_rand) for node in nodes]
    q_near = nodes[np.argmin(ndist)]
    
    q_new = {'coord': steer(q_rand, q_near['coord'], np.min(ndist), EPS)}
    
    if chk_collision([q_near['coord'], q_new['coord']], poly) == 0:
        ax.plot([q_near['coord'][0], q_new['coord'][0]], [q_near['coord'][1], q_new['coord'][1]], 'k-', linewidth=2)
        plt.pause(0.01)
        
        q_new['cost'] = dist(q_new['coord'], q_near['coord']) + q_near['cost']
        
        q_nearest = []
        r = 60
        for node in nodes:
            if chk_collision([node['coord'], q_new['coord']], poly) == 0 and dist(node['coord'], q_new['coord']) <= r:
                q_nearest.append(node)
        
        q_min = q_near
        C_min = q_new['cost']
        
        for neighbor in q_nearest:
            if dist(neighbor['coord'], q_new['coord']) + neighbor['cost'] < C_min:
                q_min = neighbor
                C_min = neighbor['cost'] + dist(neighbor['coord'], q_new['coord'])
                ax.plot([q_min['coord'][0], q_new['coord'][0]], [q_min['coord'][1], q_new['coord'][1]], 'g-')
                plt.pause(0.01)
        
        q_new['parent'] = nodes.index(q_min)
        nodes.append(q_new)
    
    result1.append(thenorm1)
    elapsed_time1.append(K1)

result_avg1 = np.mean(result1)
time_avg1 = np.mean(elapsed_time1)

D = [dist(node['coord'], q_goal['coord']) for node in nodes]

result2 = []
elapsed_time2 = []

for j in range(len(nodes)):
    K2 = np.random.rand()
    thenorm2 = np.linalg.norm(K2)
    result2.append(thenorm2)
    elapsed_time2.append(K2)

result_avg2 = np.mean(result2)
time_avg2 = np.mean(elapsed_time2)

q_final = nodes[np.argmin(D)]
q_goal['parent'] = nodes.index(q_final)
q_end = q_goal

nodes.append(q_goal)

while q_end['parent'] != 0:
    start = q_end['parent']
    ax.plot([q_end['coord'][0], nodes[start]['coord'][0]], [q_end['coord'][1], nodes[start]['coord'][1]], 'r-', linewidth=3)
    plt.pause(0.01)
    
    q_end = nodes[start]

plt.ioff()
plt.show()