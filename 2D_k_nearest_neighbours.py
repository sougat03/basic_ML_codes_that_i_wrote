import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {"purple": [[1,4], [2,3], [3,1], [1,3], [3,2]],
          "green": [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [3,3]

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNearestNeighbours: #if i put one more green point, and make k=11, the final result will always be green even if the green points are far away from out new_point, which doesnt make sense, so we keep K=3 for accuracy.

    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for catagory in self.points:
            for point in self.points[catagory]:
                distance = euclidean_distance(point , new_point)
                distances.append([distance, catagory])

        catagories = [catagory[1] for catagory in sorted(distances)[:self.k]]
        result = Counter(catagories).most_common(1)[0][0]
        return result
    
clf = KNearestNeighbours()
clf.fit(points)
print(clf.predict(new_point))

#Visualization

ax = plt.subplot()
ax.grid(True, color ="#323232")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis = 'x', color ="white")
ax.tick_params(axis='y', color = "white")

for point in points['purple']:
    ax.scatter(point[0], point[1], color ="#800080", s=60)

for point in points['green']:
    ax.scatter(point[0], point[1], color ="#00FF00", s=60)

new_class = clf.predict(new_point)
color = "#00FF00" if new_class == "red" else "#800080"
ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder = 100)

#visualising distance with dashed line
for point in points['purple']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#800080", linestyle="--", linewidth=1)

for point in points['green']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#00FF00", linestyle="--", linewidth=1)

plt.show()
