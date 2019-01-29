# -*- coding: utf-8 -*-
"""run.py: Classifies diabetes cases using Naive Bayes algorithm from scratch.
author: Willian Eduardo Becker
date: 28-03-2016
"""
import math
import random
import csv


def split_dataset(dataset, split_rate):
    train_size = int(len(dataset) * split_rate)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def load_file(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def std_dev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.iteritems():
        summaries[class_value] = summarize(instances)
    return summaries


def summarize(dataset):
    summaries = [(mean(attribute), std_dev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def get_probability(x, mean, std_dev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std_dev, 2))))
    return (1 / (math.sqrt(2*math.pi) * std_dev)) * exponent


def get_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.iteritems():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, std_dev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= get_probability(x, mean, std_dev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = get_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.iteritems():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def run():
    filename = 'diabetes.csv'
    split_rate = 0.70
    dataset = load_file(filename)
    training_set, test_set = split_dataset(dataset, split_rate)
    print('Split {0} rows into train={1} and test={2} rows').format(
        len(dataset), len(training_set), len(test_set))

    # prepare model
    summaries = summarize_by_class(training_set)

    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Model Accuracy: {0}%').format(float(accuracy))


if __name__ == "__main__":
    run()
