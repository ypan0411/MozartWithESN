import numpy as np
import csv


def create_midi(melody):
    """melody: a array with dimension n * 1"""
    n, = np.shape(melody)
    with open('sample.csv', 'a') as f:
        writer = csv.writer(f)
        i = 0
        t = 0
        while i < n:
            start = i
            if i + 1 < n:
                if melody[i + 1] == melody[i]:
                    i += 1
            i += 1
            end = i
            writer.writerow(['2', ' ' + str(start * 96), ' Note_on_c', ' 0', ' ' + str(melody[start]), ' 90'])
            writer.writerow(['2', ' ' + str(end * 96 - 4), ' Note_off_c', ' 0', ' ' + str(melody[start]), ' 0'])

        writer.writerow(['2', ' ' + str(end * 96 - 4), ' End_track'])
        writer.writerow(['0', ' 0', ' End_of_file'])


if __name__ == '__main__':
    test_melody = np.array([60, 60, 65, 66, 67, 68])
    print(np.shape(test_melody))
    create_midi(test_melody)