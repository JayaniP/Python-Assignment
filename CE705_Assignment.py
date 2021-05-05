"""Import Packages"""
import csv
import math


def load_from_csv(file_path):
    """ This function takes one parameter as filename and returns matrix(list of lists)
    :param file_path: filename
    :type file_path: string
    :return: matrix
    :rtype: list of lists
    """
    file = open(file_path)
    try:
        data = list(csv.reader(file))
        matrix = [[int(j) for j in i] for i in data]
        return matrix
    finally:
        file.close()


def get_distance(list_a, list_b):
    """ This function take two parameters as list and returns Euclidean distance between the given two lists
    :param list_a: list_a value
    :type list_a: list
    :param list_b: list_b value
    :type list_b: list
    :return: Euclidean Distance between two lists
    :rtype: int
    """
    difference = [(list_a[i] - list_b[i]) ** 2 for i in range(min(len(list_a), len(list_b)))]
    distance = math.sqrt(sum(difference))
    return distance


def get_standard_deviation(matrix, column):
    """ This function take two parameters as matrix(list of lists) and column number and returns standard deviation
    :param matrix: matrix [row] [column]
    :type matrix: list of lists
    :param column: column number
    :type column: int
    :return: standard deviation
    :rtype: int
    """
    sum_of_column = []
    sum_of_squares = []

    for row in range(len(matrix)):
        sum_of_column.append(matrix[row][column])

    column_avg = sum(sum_of_column) / len(sum_of_column)

    for row in range(len(matrix)):
        sum_of_squares.append((matrix[row][column] - column_avg) ** 2)

    standard_deviation = math.sqrt(sum(sum_of_squares) / len(sum_of_squares))
    return standard_deviation


def get_standardised_matrix(matrix):
    """ This function take one parameter as matrix(list of lists) and returns standard matrix
    :param matrix: matrix [row] [column]
    :type matrix: list of lists
    :return: matrix [row] [column]
    :rtype: list of lists
    """
    avg_array = []
    std_array = []

    for col in range(len(matrix[0])):
        sum_column = 0
        for row in range(len(matrix)):
            sum_column += matrix[row][col]
        column_avg = sum_column / len(matrix)
        avg_array.append(column_avg)
        std_array.append(get_standard_deviation(matrix, col))

    std_row = []
    std_matrix = []

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            std_value = (matrix[row][col] - avg_array[col]) / std_array[col]
            std_row.append(std_value)
        std_matrix.append(std_row)
        std_row = []
    return std_matrix


def get_k_nearest_labels(data_row, learning_data_matrix, learning_data_labels_matrix, k):
    """ This function take four parameters as row from the matrix, matrix(list of lists) from the Learning_Data.csv file,
    matrix(list of lists) from the Learning_Data_Labels.csv file and k number and returns k-nearest value of closest list
    and returns standard matrix
    :param data_row: matrix [row]
    :type data_row: list
    :param learning_data_matrix: matrix[row] [column]
    :type learning_data_matrix: list of lists
    :param learning_data_labels_matrix: matrix[row] [column]
    :type learning_data_labels_matrix: list of lists
    :param k: K-nearest value
    :type k: int
    :return: matrix
    :rtype: list of lists
    """
    euclid = []
    label_matrix = []

    for i in range(len(learning_data_matrix)):
        euclid.append(get_distance(data_row, learning_data_matrix[i]))

    k_nearest = sorted(range(len(euclid)), key=lambda x: euclid[x], reverse=False)[:k]

    for i in k_nearest:
        label_matrix.append(learning_data_labels_matrix[i])
    return label_matrix


def get_mode(label_matrix):
    """ This function take one parameter as matrix(list of lists) output of the get_k_nearest_labels() and returns mode
    of elements given in the list
    :param label_matrix: matrix[row] [column]
    :type label_matrix: list of lists
    :return: mode[frequency] of elements in list
    :rtype: int
    """
    labels = []
    for i in range(len(label_matrix)):
        labels.append(label_matrix[i][0])
    mode = max(labels, key=labels.count)
    return mode


def classify(data_matrix, learning_data_matrix, learning_data_labels_matrix, k):
    """This function take four parameters as matrix(list of lists) from the Data.csv file, matrix(list of lists) from
    the Learning_Data.csv file, matrix(list of lists) from the Learning_Data_Labels.csv file and k number and returns
    matrix(list of lists)
    :param data_matrix: matrix [row] [column]
    :type data_matrix: list of lists
    :param learning_data_matrix: matrix [row] [column]
    :type learning_data_matrix: list of lists
    :param learning_data_labels_matrix: matrix [row] [column]
    :type learning_data_labels_matrix: list of lists
    :param k: k-nearest value
    :type k: int
    :return: matrix [row] [column]
    :rtype: list of lists
    """
    std_data_matrix = get_standardised_matrix(data_matrix)
    std_learning_data_matrix = get_standardised_matrix(learning_data_matrix)
    matrix_data_labels = []

    for i in range(len(std_data_matrix)):
        matrix_data_labels.append([get_mode(get_k_nearest_labels(std_data_matrix[i], std_learning_data_matrix,
                                                                 learning_data_labels_matrix, k))])
    return matrix_data_labels


def get_accuracy(correct_data_labels_matrix, matrix_data_labels):
    """ This function take two parameters as matrix(list of lists) from Correct_Data_Labels.csv file and output of the
    classify() and returns accuracy
    :param correct_data_labels_matrix: matrix [row] [column]
    :type correct_data_labels_matrix: list of lists
    :param matrix_data_labels: matrix [row] [column]
    :type matrix_data_labels: list of lists
    :return: accuracy
    :rtype: float
    """
    if len(correct_data_labels_matrix) == len(matrix_data_labels):
        score = 0
        for i in range(len(correct_data_labels_matrix)):
            if correct_data_labels_matrix[i] == matrix_data_labels[i]:
                score = score + 1
        accuracy = score / len(correct_data_labels_matrix)
        return accuracy
    else:
        print('Data is not same')


def run_test():
    """ This function calculates accuracy of k value from 3 to 15
    :return: accuracy of given k value
    :rtype: float
    """
    data = 'Data.csv'
    correct_data_labels = 'Correct_Data_Labels.csv'
    learning_data = 'Learning_Data.csv'
    learning_data_labels = 'Learning_Data_Labels.csv'

    correct_data_labels_matrix = load_from_csv(correct_data_labels)
    learning_data_matrix = load_from_csv(learning_data)
    learning_data_labels_matrix = load_from_csv(learning_data_labels)

    std_data_matrix = get_standardised_matrix(load_from_csv(data))
    std_learning_data_matrix = get_standardised_matrix(learning_data_matrix)

    for i in range(3, 16):
        data_labels_matrix = classify(std_data_matrix, std_learning_data_matrix, learning_data_labels_matrix, i)
        accuracy = round(get_accuracy(correct_data_labels_matrix, data_labels_matrix), 2)
        print('k=' + str(i) + ', Accuracy = ' + str(accuracy * 100) + '%')


######  OUTPUT #######
# k=3, Accuracy = 95.0%
# k=4, Accuracy = 95.0%
# k=5, Accuracy = 96.0%
# k=6, Accuracy = 96.0%
# k=7, Accuracy = 94.0%
# k=8, Accuracy = 96.0%
# k=9, Accuracy = 96.0%
# k=10, Accuracy = 96.0%
# k=11, Accuracy = 96.0%
# k=12, Accuracy = 96.0%
# k=13, Accuracy = 96.0%
# k=14, Accuracy = 96.0%
# k=15, Accuracy = 95.0%