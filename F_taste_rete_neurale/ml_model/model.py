
from F_taste_rete_neurale.config.definitions import ROOT_DIR
import numpy as np
import os
import torch
import torch.nn as nn


class SingleLayerFeedforwardNN(nn.Module):
    def __init__(self, input_size=None, output_size=None):
        if input_size is not None and output_size is not None:
            super(SingleLayerFeedforwardNN, self).__init__()
            # Definisce il layer nascosto con 12 nodi. Assumi input_size e output_size come dimensioni di input e output.
            self.hidden = nn.Linear(input_size, 12)  # 10 nodi nel layer nascosto
            # Definisce la funzione di attivazione ReLU per il layer nascosto
            self.relu = nn.ReLU()
            # Definisce il layer di output
            self.output = nn.Linear(12, output_size)  # output_size è la dimensione dell'output desiderata
            # Definisce la funzione di attivazione sigmoide per il layer di output
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Passa i dati attraverso il layer nascosto, poi applica la ReLU
        x = self.relu(self.hidden(x))
        # Passa i dati attraverso il layer di output, poi applica la sigmoide
        x = self.sigmoid(self.output(x))
        return x
    



def get_mean_and_std():
    std_path = os.path.realpath(os.path.join(ROOT_DIR,"resources","machine_learning_model","mean_std","std.npy"))
    mean_path = os.path.realpath(os.path.join(ROOT_DIR,"resources","machine_learning_model","mean_std","mean.npy"))

    return np.load(mean_path), np.load(std_path)


def pre_process_data(features_vector: list):
    if len(features_vector) != 9:
        raise Exception("Wrong number of features")

    # convert list to numpy array
    features_vector = np.array(features_vector, ndmin=2)

    mean, std = get_mean_and_std()


    # data standardization
    for i, feature in enumerate(features_vector):
        feature = (feature - mean[i]) / std[i]

    return features_vector


def predict(data):
    
    model = SingleLayerFeedforwardNN(input_size=9, output_size=1)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR,"resources","machine_learning_model","f_taste_neural_net.pt")))
    model.eval()

    output = model(torch.tensor(data, dtype=torch.float32))

    return output.item()
    #filename = os.path.join(ROOT_DIR,"resources","machine_learning_model","model")
    #model = tf.keras.Sequential([
    #    hub.KerasLayer(filename)
    #])
    #model.build(data.shape)
    #return str(model.predict(data))
    #session = onnxruntime.InferenceSession(os.path.join(ROOT_DIR,"resources","machine_learning_model","model.onnx"))
    #input_name = session.get_inputs()[0].name
    #outputs_name = session.get_outputs()[0].name
    #model_meta = session.get_modelmeta()
    #outputs = session.run([outputs_name], {input_name: data.astype(np.float32)})
    #return str(outputs[0][0][0])