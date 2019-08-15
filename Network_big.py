import numpy as np


class neuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, layers):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = layers[0]
        self.hnodes = layers[1]
        self.h2nodes = layers[2]
        self.onodes = layers[3]

        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.whh = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.whh = (np.random.rand(self.hnodes, self.h2nodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # использование сигмоиды в качестве функции активации
        self.activation_func = lambda x: 1.0 / (1.0 + np.exp(-x))

    def save(self, wih, whh, who):
        np.save(wih, self.wih)
        np.save(whh, self.whh)
        np.save(who, self.who)

    def load(self, wih, whh, who):
        self.wih = np.load(wih)
        self.whh = np.load(whh)
        self.who = np.load(who)
        self.inodes = self.wih[0]
        self.hnodes = self.wih[1]
        self.h2nodes = self.whh[1]
        self.onodes = self.who[0]

    # тренировка нейронной сети
    def train(self, input_list, target, lr):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        hidden2_inputs = np.dot(self.whh, hidden_outputs)
        hidden2_outputs = self.activation_func(hidden2_inputs)

        final_inputs = np.dot(self.who, hidden2_outputs)
        final_outputs = self.activation_func(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        # print("истинное значение - {0} полученное {1}".format(np.argmax(targets), np.argmax(final_outputs)))
        hidden2_errors = np.dot(self.who.T, output_errors)

        hidden_errors = np.dot((self.whh.T, hidden2_errors))

        # обновить весовые коэффициенты связей между выходным и скрытым слоями
        self.who += lr * np.dot((output_errors * final_outputs *
                                 (1.0 - final_outputs)),
                                np.transpose(hidden2_outputs))
        # обновить весовые коэффициенты связей между скрытыми слоями
        self.whh += lr * np.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
                                np.transpose(hidden_outputs))

        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += lr * np.dot((hidden_errors * hidden_outputs *
                                 (1.0 - hidden_outputs)),
                                np.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        # return final_outputs
        return np.argmax(final_outputs)

    def learn(self, training_set, test_set, epochs_num, learn_rate):

        for i in range(1, epochs_num + 1):
            # foreach epoch
            index = 0
            # обучение
            for record in training_set:
                # print(index)
                index += 1
                # foreach symbol in set
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]))
                targets = np.zeros(n.onodes) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets, learn_rate)
            print("epoch {0}".format(i))

            # контроль
            scorecard = []  # Список ответов сети на контрольный сет
            label = []  # Список правильных ответов
            for record in test_set:
                all_values = record.split(',')
                correct = all_values[0]
                label.append(all_values[0])

                inputs = (np.asfarray(all_values[1:]))
                # targets = np.zeros(n.onodes) + 0.01
                # targets[int(all_values[0])] = 0.99

                result = self.query(inputs)

                if int(result) == int(correct):
                    scorecard.append(1)
                else:
                    scorecard.append(0)

            # генерация отчета о тестовом прогоне
            score = np.array(scorecard)

            # print(score)
            # print(label)
            print(" Эффективность = ", (score.sum() / score.size))


if __name__ == "__main__":
    nodes = [400, 200,200, 159]
    epochs = 100
    learning_rate = 0.3

    n = neuralNetwork(nodes)
    # загрузить в список тренировочный набор данных CSV-файла
    training_data_file = open("training_dataset_big.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # загрузить в список тестовый набор данных CSV-файла
    test_data_file = open("test_dataset_big.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # тренировка нейронной сети

    n.learn(training_data_list, test_data_list, epochs, learning_rate)
    n.save("wih", "who")
