"""Funciones relacionadas con el calculo y visualizacion de un SOM"""
import os, pickle, numpy as np
from minisom import MiniSom
from numpy import argsort, array, mean
from matplotlib.patches import RegularPolygon
from matplotlib import cm, colorbar, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import AgglomerativeClustering


"""metodo para entrenar el som"""
def train_som (data, dim, steps, dir):
    #entrenamos el modelo
    sigma=2.9; learning_rate=.7; activation_distance='euclidean'; topology='hexagonal'; neighborhood_function='gaussian'
    np.random.seed(0)
    modelFile = dir+ 'somModel.pkl'
    if not os.path.exists(modelFile):
        som = MiniSom(dim[0], dim[1], data.shape[1], sigma=sigma, learning_rate=learning_rate, activation_distance=activation_distance,
                  topology=topology, neighborhood_function=neighborhood_function, random_seed=10)
        som.train(data, steps, verbose=True)
        with open(modelFile, 'wb') as f: pickle.dump(som, f)
    else:
        with open(modelFile, 'rb') as f: som = pickle.load(f)

    #guardamos los errores del modelo
    with open(dir+ 'error_somModel.txt', 'w') as f:
        f.write('Quantization error: '+str(som.quantization_error(data))+'\n')
        f.write('Topographic error: ' + str(topographic_error(som, data))+'\n')
    #som_functions.draw_errors_som(data, dim, steps, dir, sigma, learning_rate, activation_distance, topology, neighborhood_function)
    return som

"""metodo clusterizar el som. obtiene los pesos,los cluseriza y mira
 cada dato a que neurona va, para asignarle ese cluster"""
def cluster_som(som, data, numClusters):
    np.random.seed(0)
    weights = np.transpose(som.get_weights(), (1, 0, 2))
    selectedNeuron = np.array(list(map(som.winner, data)))
    selectedNeuron = weights.shape[1] * selectedNeuron[:, 1] + selectedNeuron[:, 0]
    weights = weights.reshape((weights.shape[0] * weights.shape[1], weights.shape[2]))
    clusters = AgglomerativeClustering(numClusters).fit_predict(weights)
    return clusters[selectedNeuron]


"""Return the topographic error for hexagonal grid"""
def topographic_error(som, data):
    b2mu_inds = argsort(som._distance_from_weights(data), axis=1)[:, :2]
    b2mu_coords = [[get_euclidean_coordinates_from_index(som,bmu[0]),
                        get_euclidean_coordinates_from_index(som,bmu[1])]
                       for bmu in b2mu_inds]
    b2mu_coords = array(b2mu_coords)
    b2mu_neighbors = [(bmu1 >= bmu2 - 1) & ((bmu1 <= bmu2 + 1))
                          for bmu1, bmu2 in b2mu_coords]
    b2mu_neighbors = [neighbors.prod() for neighbors in b2mu_neighbors]
    te = 1 - mean(b2mu_neighbors)
    return te

"""Returns the Euclidean coordinated of a neuron using its index as the input"""
def get_euclidean_coordinates_from_index(som, index):
    if index < 0:
        return (-1, -1)
    y = som._weights.shape[1]
    coords = som.convert_map_to_euclidean((int(index/y), index % y))
    return coords

"""Dibuja los resultados de clasificacion en el som graficamente"""
def draw_som(som, data, classification, dirOutExp):
    #definimos los marcadores de la figura, para mas de 6 clusters hay que añadir mas
    markers = ['o', '+', 'x', '*', '.', 's']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    #definimos la figura
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')

    # creamos el fondo de la grafica
    xx, yy = som.get_euclidean_coordinates()
    weights = som.get_weights()
    umatrix = som.distance_map()
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
            ax.add_patch(hex)

    #ponemos las marcas de la clasificacion de la grafica
    for cnt, x in enumerate(data):
        # getting the winner
        w = som.winner(x)
        # place a marker on the winning position for the sample xx
        wx, wy = som.convert_map_to_euclidean(w)
        wy = wy * np.sqrt(3) / 2
        plt.plot(wx, wy, markers[classification[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[classification[cnt]], markersize=12, markeredgewidth=2)

    #configuramos los bordes y demas elementos de la grafica
    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange - .5, xrange)
    plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('distance from neurons in the neighbourhood', rotation=270, fontsize=16)
    plt.gcf().add_axes(ax_cb)
    plt.savefig(dirOutExp+'/'+str(classification.max()+1)+'_clusters_som_classification.png')
    plt.close()

"""dibuja el error del som"""
def draw_errors_som(data, dim, steps, dir, sigma, learning_rate, activation_distance, topology, neighborhood_function):
    np.random.seed(0)
    som = MiniSom(dim[0], dim[1], data.shape[1], sigma=sigma, learning_rate=learning_rate, activation_distance=activation_distance,
                  topology=topology, neighborhood_function=neighborhood_function, random_seed=10)
    q_error = [];t_error = []
    for i in range(steps):
        rand_i = np.random.randint(len(data))
        som.update(data[rand_i], som.winner(data[rand_i]), i, steps)
        q_error.append(som.quantization_error(data))
        t_error.append(topographic_error(som, data))

    plt.plot(np.arange(steps), q_error, label='quantization error')
    plt.plot(np.arange(steps), t_error, label='topographic error')
    plt.ylabel('error')
    plt.xlabel('iteration index')
    plt.legend()
    plt.savefig(dir + '/error_somModel.png')
    plt.close()