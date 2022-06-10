from utils_functions import *
from plan import *
from collections import Counter


dizionario_stati = load_file("./dizionario_stati")
piani_caricati = load_file("./plans")


def costruisci_vettore(dizionario, numero_piano):
    lista_stati = piani_caricati[numero_piano].states
    lunghezza_dizionario = len(dizionario)
    vettore_stati = []
    for stato in lista_stati:
        vettore = [0] * lunghezza_dizionario    # per ciascuno stato crea un vettore di zeri lungo come il dizionario
        for s in stato:
            for key in dizionario.keys():
                if key == s:
                    vettore[dizionario[key] - 1] = 1
        vettore_stati.append(vettore)
    return vettore_stati


def costruisci_vettore_iniziale(dizionario, numero_piano):
    stato_iniziale = piani_caricati[numero_piano].initial_state
    lunghezza_dizionario = len(dizionario)
    vettore_iniziale = [0] * lunghezza_dizionario
    for s in stato_iniziale:
        for key in dizionario:
            if key == s:
                vettore_iniziale[dizionario[key] - 1] = 1
    return vettore_iniziale


def costruisci_vettore_goal(dizionario, numero_piano):
    stato_goal = piani_caricati[numero_piano].goals
    lunghezza_dizionario = len(dizionario)
    vettore_goal = [0] * lunghezza_dizionario
    for s in stato_goal:
        for key in dizionario:
            if key == s:
                vettore_goal[dizionario[key] - 1] = 1
    return vettore_goal


def stampa_info_piano(numero_piano):
    print("***")
    print("")
    print("Piano numero:", numero_piano)
    print("")
    print("***")
    print("")
    print("Stato iniziale del piano", numero_piano, "->")
    print(piani_caricati[numero_piano].initial_state)
    print("Lunghezza vettore stato iniziale =", len(piani_caricati[numero_piano].initial_state))
    print("")
    print("***")
    print("")
    print("Stati intermedi del piano", numero_piano, "->")
    i = 1
    for p in piani_caricati[numero_piano].states:
        print(i, ": ", p)
        print("lunghezza vettore", i, "=", len(p))
        i += 1
    print("")
    print("***")
    print("")
    print("Stato goal del piano", numero_piano, "->")
    print(piani_caricati[numero_piano].goals)
    print("Lunghezza vettore stato goal =", len(piani_caricati[1].goals))
    print("")
    print("***")


def stampa_info_vettore_stati(vettore, numero_piano):
    print("")
    print("***")
    print("")
    print("Lunghezza del vettore degli stati:", len(vettore))
    print("")
    print("***")
    print("")
    for v in vettore:
        print(v)
        counts = Counter(v)
        print(counts[1], "uni su", len(v), "zeri")


def stampa_info_vettore_iniziale(vettore, numero_piano):
    print("")
    print("***")
    print("")
    print("Lunghezza del vettore iniziale:", len(vettore))
    print("")
    print("***")
    print("")
    print(vettore)
    counts = Counter(vettore)
    print(counts[1], "uni su", len(vettore), "zeri")


def stampa_info_vettore_goal(vettore, numero_piano):
    print("")
    print("***")
    print("")
    print("Lunghezza del vettore goal:", len(vettore))
    print("")
    print("***")
    print("")
    print(vettore)
    counts = Counter(vettore)
    print(counts[1], "uni su", len(vettore), "zeri")


# Non utilizzare adesso, va ottimizzata build_vector prima, sia a livello di complessità,
# sia a livello di codifica di "1" e "0", e strttura dati da usare
def build_all_vectors(dict, plans_list):
    total = []
    for plan in plans_list:
        r = costruisci_vettore(dict, plan.states)
        total.append(r)
    return total


stampa_info_piano(1)

vettore_prova_stati = costruisci_vettore(dizionario_stati, 1)
vettore_prova_iniziale = costruisci_vettore_iniziale(dizionario_stati, 1)
vettore_prova_goal = costruisci_vettore_goal(dizionario_stati, 1)

stampa_info_vettore_stati(vettore_prova_stati, 1)
stampa_info_vettore_iniziale(vettore_prova_iniziale, 1)
stampa_info_vettore_goal(vettore_prova_goal, 1)
