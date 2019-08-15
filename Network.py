import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

class neuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes=400, hiddennodes=200, outputnodes=26):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # использование сигмоиды в качестве функции активации
        self.activation_func = lambda x: 1.0 / (1.0 + np.exp(-x))

    def load(self, wih_file, who_file):
        self.wih = np.load(wih_file)
        self.who = np.load(who_file)
        self.inodes = self.wih[0]
        self.hnodes = self.wih[1]
        self.onodes = self.who[0]

    def save(self, wih_path, who_path):
        np.save(wih_path, self.wih)
        np.save(who_path, self.who)

    # тренировка нейронной сети
    def train(self, input_list, target, learn_rate):
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя

        # print(self.wih)
        # print(inputs)
        # a_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(numpy.float32))
        hidden_inputs = gpuarray.to_gpu(np.dot(self.wih, inputs))

        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_func(hidden_inputs.get())
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = gpuarray.to_gpu(np.dot(self.who, hidden_outputs))
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_func(final_inputs.get())
        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.who += learn_rate * np.dot((output_errors * final_outputs * \
                                         (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += learn_rate * np.dot((hidden_errors * hidden_outputs * \
                                         (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = gpuarray.to_gpu(np.dot(self.wih, inputs))
        hidden_outputs = self.activation_func(hidden_inputs.get())
        final_inputs = gpuarray.to_gpu(np.dot(self.who, hidden_outputs))
        final_outputs = self.activation_func(final_inputs.get())
        # return final_outputs
        return chr(final_outputs.argmax() + 97)

    def learn(self, training_set, test_set, learn_rate, epochs, rand=False):
        from time import time

        scorecard = []

        # перебрать все записи в тренировочном наборе данных
        for i in range(1, epochs + 1):
            start = time()
            data = training_set
            if rand: np.random.shuffle(data)
            for record in data:
                # получить список значений, используя символы запятой (',')
                # в качестве разделителей
                all_values = record.split(',')
                # масштабировать и сместить входные значения
                inputs = (np.asfarray(all_values[1:]))
                # создать целевые выходные значения (все равны 0,01, за исключением
                # желаемого маркерного значения, равного 0,99)
                targets = np.zeros(self.onodes) + 0.01

                # print("targets")
                # print(targets)
                # all_values[0] - целевое маркерное значение для данной записи
                targets[ord(all_values[0]) - 97] = 0.99
                n.train(inputs, targets, learn_rate)
            print("Эпоха № {0}".format(i))
            finish = time()
            print("Время эпохи: " + str(finish - start))

            testdata = test_set
            if rand: np.random.shuffle(testdata)
            for record in testdata:
                all_values = record.split(',')
                correct = all_values[0]
                # масштабировать и сместить входные значения
                inputs = (np.asfarray(all_values[1:]))
                # создать целевые выходные значения (все равны 0,01, за исключением
                # желаемого маркерного значения, равного 0,99)
                targets = np.zeros(self.onodes) + 0.01

                # print("targets")
                # print(targets)
                # all_values[0] - целевое маркерное значение для данной записи
                targets[ord(all_values[0]) - 97] = 0.99
                result = n.query(inputs)
                if result == correct:
                    scorecard.append(1)
                else:
                    scorecard.append(0)

            score = np.array(scorecard)
            label = []
            for i in testdata:
                label.append(i[0])
            print(label)
            print(" Точность сети = ", (score.sum() / score.size))



if __name__ == "__main__":

    learning_rate = 0.5
    epochs = 150
    # создать экземпляр нейронной сети
    n = neuralNetwork()
    #n.load("wih.npy", "who.npy")
    training_data_file = open("training_dataset_times.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # загрузить в список тестовый набор данных CSV-файла набора MNIST
    test_data_file = open("test_dataset_times.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # тренировка нейронной сети
    n.learn(training_data_list, test_data_list, learning_rate, epochs)
    n.save("wih", "who")

