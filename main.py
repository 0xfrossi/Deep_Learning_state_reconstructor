import random
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from utils_functions import *

"""
Caricamento dei dizionari e dei piani
"""


dizionario_stati = load_file("./dizionario_stati")
piani_caricati = load_file("./plans")


"""
Definizione dei metodi di costruzione dei vettori da dare in pasto all'autoencoder
"""


def costruisci_vettore(dizionario, lista_stati):
    lunghezza_dizionario = len(dizionario)
    vettore_stati = []
    for stato in lista_stati:
        vettore = [0] * np.array([0]*lunghezza_dizionario, dtype=np.int8)
        for s in stato:
            for key in dizionario.keys():
                if key == s:
                    vettore[dizionario[key] - 1] = 1
                    break
        t = tf.convert_to_tensor(vettore, dtype=tf.int8)
        vettore_stati.append(t)
    r = tf.convert_to_tensor(vettore_stati, dtype=None)
    return vettore_stati


# NON UTILIZZATO ADESSO
# shape di ogni singolo elemento (r) è (n x 340) con n che varia su ogni piano, raggruppati sulla base dei piani
def costruisci_tutti_vettori(dizionario, lista_piani):
    total = []
    for piano in lista_piani:
        r = costruisci_vettore(dizionario, piano.states)
        total.append(r)
    # r = tf.convert_to_tensor(total,dtype=None)
    return total


# Gli stati vengono ordinati (in una lista) singolarmente con shape (1x340), non vengono raggruppati sulla base dei piani
# Da utilizzare per autoencoder standard
def costruisci_tutti_vettori_1x340(dizionario, lista_piani):
    lunghezza_dizionario = len(dizionario)
    total = []
    for plan in lista_piani:
        for stato in plan.states:
            vettore = np.array([0]*lunghezza_dizionario, dtype=np.int8)
            for s in stato:
                for key in dizionario.keys():
                    if key == s:
                        vettore[dizionario[key]-1] = 1
                        break
            t = tf.convert_to_tensor(vettore, dtype=tf.int8)
            total.append(t)
    return total


"""
Preparazione dei dati
"""


def crea_set(dizionario, piani):
    s_training = load_file("./set_training")
    s_test = load_file("./set_test")
    s_validation = load_file("./set_validation")

    # se per qualche motivo almeno uno manca, li rifaccio tutti
    if s_training is None or s_test is None or s_validation is None:
        # Costruzione del dataset completo -> lo faccio solo se serve, non se ho già i file da caricare
        tutti_1x340 = costruisci_tutti_vettori_1x340(dizionario, piani)
        random.shuffle(tutti_1x340)
        # Suddivisione in Train, Test e Validation sets. Input shape per la rete = 340
        split = int(len(tutti_1x340)//3)
        train_set = tutti_1x340[split:]
        vt = tutti_1x340[:split]
        sub_split = int(len(vt)//2)
        validation_set = vt[sub_split:]
        test_set = vt[:sub_split]
        save_file(train_set, ".", "set_training")
        save_file(validation_set, ".", "set_validation")
        save_file(test_set, ".", "set_test")


crea_set(dizionario_stati, piani_caricati)

s_training = load_file("./set_training")
s_test = load_file("./set_test")
s_validation = load_file("./set_validation")

"""
Modello
"""

# Costante
input_size = 340
# Scelta casuale -> fare dei test
hidden_size = 85
code_size = 20


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.TensorBoard(log_dir="./logsTf")
]


# Cose da fare per migliorare la rete solo dopo aver fatto i primi tentativi con la rete proposta:
# * Provare ad aggiungere Regolarizzazione es. L1,L2 e dropout(solo nella fase di encoding)
# * Provare swish al posto di relu
# * Provare keras.layers.BatchNormalization()
# * Fare Hyperparameter Tuning  (ultima) www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# Salvare i risultati con ogni modifica fatta per scriverli nel report

input_layer = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu', kernel_initializer="he_uniform")(input_layer)
code = Dense(code_size, activation='relu', kernel_initializer="he_uniform")(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu', kernel_initializer="he_uniform")(code)
output_layer = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_layer, output_layer)
# variare learning_rate beta_1,beta_2, da fare per ultimo
# batch_size ???
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=["accuracy", "Precision", "Recall"])
autoencoder.fit(x=s_training, y=s_training, epochs=50, batch_size=10000, validation_data=(s_validation, s_validation), callbacks=my_callbacks, verbose=1)

