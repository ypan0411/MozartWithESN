import numpy as np
import matplotlib.pyplot as plt


def printall(pianoroll_r, pianoroll_l, T, T1):
    for i in range(5):
        plt.plot(pianoroll_r[:T, i], 'bs')
        plt.plot(pianoroll_l[:T, i], 'rs')

    plt.axis((0, T1, 24, 96))
    plt.show()

def printmelody(melody):
  plt.plot(melody, 'bs')
  plt.axis((0, 320, 24, 96))  # From C1 to C7
  plt.show()