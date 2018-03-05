import numpy as np
import csv


def read_matrix(name):

    csv_reader = csv.reader(open(name), delimiter=' ')

    compressed_matrix = []
    for row in csv_reader:
        compressed_matrix.append(row)

    label_cols = np.zeros(len(compressed_matrix))
    data = []
    row = []
    col = []
    idx = 0
    idy = 0
    
    for line in compressed_matrix:
        label_cols[idx] = int(line[0])
        for j in range(1, len(line)):
            tmp = line[j].split(':')
            if len(tmp)==2:
                data.append(float(tmp[1]))
                row.append(idx)
                col.append(int(tmp[0])-1)
                idy = idy+1
        idx = idx+1

    return data, row, col, label_cols

def read_data(name):
    csv_reader = csv.reader(open(name), delimiter=' ')

    compressed_matrix = []
    for row in csv_reader:
        compressed_matrix.append(row)

    data = []
    row = []
    col = []
    idx = 0
    idy = 0

    for line in compressed_matrix:
        for j in range(len(line)-1):
            tmp = line[j].split(':')
            data.append(int(tmp[1]))
            row.append(idx)
            col.append(int(tmp[0])-1)
            idy = idy+1
        idx = idx+1

    return data, row, col

def read_pos(name):
    csv_reader = csv.reader(open(name), delimiter=' ')

    compressed_matrix = []
    for row in csv_reader:
        compressed_matrix.append(row[:-1])

    data = []
    row = []
    col = []
    idx = 0
    idy = 0
    for line in compressed_matrix:
        for j in line:
            data.append(1)
            row.append(idx)
            col.append(int(j)-1)
        idx = idx+1

    return data, row, col