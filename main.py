from math import exp

import numpy
import matplotlib.pyplot as plt


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


def act(action, Qstar):
    return numpy.random.normal(Qstar[action], 1)


def run(epsilon):
    history = [0 for i in range(1000)]

    for task in range(2000):
        Qstar = [numpy.random.normal(0, 1) for i in range(10)]
        Q = [0 for i in range(10)]
        counts = [0 for i in range(10)]
        for t in range(1, 1001):
            if numpy.random.randint(0, 100) < epsilon*100:
                action = numpy.random.randint(0, len(Q))
            else:
                averages = [(Q[i] / counts[i] if counts[i] > 0 else 0) for i in range(10)]
                action = averages.index(max(averages))

            reward = act(action, Qstar)
            Q[action] += reward
            counts[action] += 1

            history[t-1] += reward

    return [elem/2000 for elem in history]


def runSoftmax(temperature):
    history = [0 for i in range(1000)]

    for task in range(2000):
        Qstar = [numpy.random.normal(0, 1) for i in range(10)]
        Q = [0 for i in range(10)]
        for t in range(1, 1001):
            averages = [q / t for q in Q]

            probabilities = softmax([q/temperature for q in averages])
            action = numpy.random.choice(10, p=probabilities)

            reward = act(action, Qstar)
            Q[action] += reward

            history[t - 1] += reward

    return [elem / 2000 for elem in history]

if __name__ == '__main__':
    plt.plot(run(0.1), 'b', label="ɛ=0.1")
    plt.plot(run(0.01), 'r', label="ɛ=0.01")
    plt.plot(run(0), 'g', label="ɛ=0")
    plt.xlabel('Plays')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    plt.plot(runSoftmax(1), 'b', label="τ=1")
    plt.plot(runSoftmax(0.1), 'r', label="τ=0.1")
    plt.plot(runSoftmax(0.01), 'g', label="τ=0.01")
    plt.xlabel('Plays')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
