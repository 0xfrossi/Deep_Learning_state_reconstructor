import random
import datetime, os
from utils_functions import *
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path  
import traceback

"""
Caricamento dei dizionari e dei piani
"""


#dizionario_stati = load_file("./dizionario_stati")
#piani_caricati = load_file("./plans")


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
        vettore_stati.append(vettore)
    r = np.array(vettore_stati)
    return r


# NON UTILIZZATO ADESSO
# shape di ogni singolo elemento (r) è (n x 340) con n che varia su ogni piano, raggruppati sulla base dei piani
def costruisci_tutti_vettori(dizionario, lista_piani):
    total = []
    for piano in lista_piani:
        r = costruisci_vettore(dizionario, piano.states)
        total.append(r)
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
            total.append(vettore)
    total = np.array(total)
    return total


"""
Preparazione dei dati
"""


def crea_set(dizionario, piani, s_tr, s_te, s_va):
    # s_training = load_file("./set_training")
    # s_test = load_file("./set_test")
    # s_validation = load_file("./set_validation")

    # se per qualche motivo almeno uno manca, li rifaccio tutti
    if s_tr is None or s_te is None or s_va is None:
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
        return True
    else:
        return False

#confronta posizione per posizione se i vettori sono uguali, se sono diversi errore++
def check_singolo(input, decoded):
    errore=0
    decoded=np.round(decoded)
    for i in range(len(input)):
        if input[i]!=decoded[i]:
            errore +=1
    return errore

#applica check_singolo per ogni vettore degli insiemi
def compute_all_errors(inputSet,decodedSet):
    l=len(inputSet)
    errori_set=[]
    for j in range(l):
      n_errori= check_singolo(inputSet[j],decodedSet[j]) 
      errori_set.append(n_errori)
    errori_set= np.array(errori_set,dtype=np.int32) 
    media= np.mean(errori_set) 
    return  errori_set, media


def occorrenze(arr):
    occ=np.bincount(arr)
    # errore : occorrenza 
    return dict(zip(arr,occ))


    
#Aggiungi altre info utili in caso
#input: inputSet- set originale per encoded, decoded- set encoded e decoded, dir- directory di tensorboard attuale
#Restituisce info utili sui risultati della procedura di ae sul dataset provato, quali:
# * num di vettori encoded e decoded correttamente
# * num di valori non decofificati correttamente (errori) in un vettore, raggruppati in una lista per ogni vettore
# * png, distribuzione del numero di errori per vettore su tutto il set in esame

def results_info(inputSet,decodedSet,dir):
    OCC="Occorrenze-  errori (elem della lista) : occorrenza\n"
    #array_errori= lista che contiene il num di errori per ogni vettore del set in esame, errore= non corrispondenza tra 0 e 1
    #ogni elemento della lista (errore) è calcolaro tramite 'check_singolo()'
    array_errori, media=compute_all_errors(inputSet,decodedSet) 
    l=len(inputSet)       
    no=0
    for i in array_errori:
        if i==0:
            no+=1

    risultato="Numero di vettori decodificati correttamente: {0} su {1} totali \nRapporto: {2:.3f}% \nMedia num di errori per array: {3:.3f}\n".format(no, l, (no/l)*100 ,media)        
    print(risultato)

    occorr=occorrenze(array_errori)

    plt.hist(array_errori)
    plt.xlabel("No of errors")
    plt.ylabel("No of samples")
    plt.show()
    with open(dir+'/infoTest.txt', 'w') as f:
        f.writelines([risultato,OCC,str(occorr)])
        f.close()

    plt.savefig(dir+'/distribuzione_errori.png',format='png')

    return array_errori  



