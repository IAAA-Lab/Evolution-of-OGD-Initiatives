"""Funciones relacionasdas con la carga y guardadod de datos"""
import numpy as np, pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

"""carga y limpia los datos"""
def loadAndCleanModel(file, mode ='normal'):
    np.random.seed(0)
    pandas.set_option('display.max_columns', None);
    data = pandas.DataFrame();
    xl = pandas.ExcelFile(file);
    for sheet in xl.sheet_names:
        df = xl.parse(sheet);
        df.insert(2, "year", int(sheet.split("_")[0]));
        df = df.dropna(how='any', axis=0);
        df1 = df.iloc[:, :5];
        df2 = df.iloc[:, 5:];
        df2 = df2-df2.mean();
        df1 = df1.join(df2);
        data = data.append(df1, ignore_index = True);
    scaler = MinMaxScaler()

    if mode == 'ext' :
        data.insert(3, 'CAGROR', data['CAGR'])
        data.iloc[:, 4:] = scaler.fit_transform(data.iloc[:, 4:].to_numpy())
        if (data['year'] == 1990).any():
            data['year'] += 1
    else:
        data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:].to_numpy())

    return shuffle(data);

"""Guarda los datos originales con la classificacion obtenida"""
def saveSomClassification(data, neuron, classification, dirOutExp):
    df = data.copy()
    df.insert(0, 'cluster', classification)
    df.insert(1, 'n fila', neuron[:,1])
    df.insert(2, 'n columna', neuron[:,0])
    df = df.sort_values(['id', 'year'])
    df.to_excel(dirOutExp + '/' + str(classification.max() + 1) + '_clusters_resultado.xlsx', index=False)

#funcion que guarda los resultados de un experimento de entrenamiento con un modelo en un fichero
def saveResultToFile (dirOutExp, inFile, fil, expType, modelName, clasif_stats, stat_tests, accuracy, confMat, bestParam =''):
    with open(dirOutExp + expType+'_'+modelName+'_resultados.txt', 'w') as f:
        f.write('Fichero: ' + inFile + '\nModo: ' + fil + '\n')
        f.write('Datos en las categorias:\n' + str(clasif_stats) + '\n')
        f.write('parametros: ' + str(bestParam) + '\n')
        f.write('Precisión: ' + str(accuracy) + '\n')
        f.write('Datos en las categorias de los tests:\n' + str(stat_tests) + '\n')
        f.write('MatrizConfusion:\n')
        f.write(np.array2string(confMat, separator=' ') + '\n')

