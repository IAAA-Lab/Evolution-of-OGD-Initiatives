"""Herramienta de clustering usando som sobre datos de población"""
import itertools
import numpy as np
import os
import pandas

from draw_functions import draw_tsne, draw_pca, draw_dendogram
from load_save_functions import loadAndCleanModel, saveSomClassification
from som_functions import train_som, draw_som, cluster_som

dirIn = 'data/input/New Experiments/'
dirOut= 'data/output/'

"""main del programa"""
som_dimensions = [7,8]
steps = 100000
clusters = [6]


inFile = "Exp 3(a)Retry6Clusters.xlsx"

if __name__ == '__main__':
    dirOutExp=dirOut+inFile+"_"+str(som_dimensions[0])+"_"+str(som_dimensions[1])+"_"+str(steps)+"/"

    if not os.path.exists(dirOutExp): os.makedirs(dirOutExp)

    datac= loadAndCleanModel(dirIn+inFile)

    data = datac


    data_subset = data.iloc[:, 3:].values
    print(data_subset)

    #analizamos las caracteristicas de los datos
    draw_dendogram(data_subset, dirOutExp)

    #para los datos completos dibujamos una grafica en la que se muestren coloreados los considerados
    considered = data["considered"].to_numpy()
    #draw_tsne(data_subset, considered, dirOutExp, "considered");
    draw_pca(data_subset, considered, dirOutExp, "considered");

    #creamos el som y generamos y guardamos primero los pesos
    som = train_som(data_subset, som_dimensions, steps, dirOutExp)
    centroid = som.get_weights().reshape(som_dimensions[0]*som_dimensions[1],-1);
    indices = np.indices(som_dimensions).reshape(2, -1).T[:,[1,0]]
    result = np.concatenate((indices, centroid), axis=1)
    df2 = pandas.DataFrame(result)
    df2.columns = data.columns[3:].insert(0, 'column').insert(0, 'row')
    df2 = df2.sort_values(by=['row', 'column'])
    df2.to_excel(dirOutExp + 'som_weights.xlsx', index=False)

    #guardamos la clasificacion para cada numero de clusters
    for c in clusters:
        dataClassification = cluster_som(som, data_subset, c)
        #draw_tsne(data_subset, dataClassification, dirOutExp);
        draw_pca(data_subset, dataClassification, dirOutExp);
        draw_som(som, data_subset, dataClassification, dirOutExp)
        saveSomClassification(data, np.array(list(map(som.winner, data_subset))), dataClassification, dirOutExp)









