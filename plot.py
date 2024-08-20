import matplotlib.pyplot as plt

probs = []
with open("/Users/luoyichao/Desktop/prediction.txt", "r") as f:
    for line in f.readlines():
        probs.append(float(line.split(',')[-1]))
probs = sorted(probs)
plt.plot([i for i in range(len(probs))], probs)
plt.show()
