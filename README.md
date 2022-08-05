# Deep_Learning_Autoencoder

 [AutoencoderBase.ipnb](AutoencoderBase.ipynb) Notebook in cui si trova il primo task di encoding base.

 [RicostruttoreVettore.ipynb](RicostruttoreVettore.ipynb) Notebook contenente il secondo task di ricostruzione di un vettore (centrale) data una sequenza di vettori, due varianti:

1. Data una sequenza di 5 vettori di cui 4 in input, ricostruire il quinto (quello centrale nella sequenza).
2. Data una sequenza di 7 vettori di cui 6 in input, ricostruire il settimo (quello centrale nella sequenza).

[utils.py](utils.py) File contenente funzioni di servizio.

[TestLogs](TestLogs) Cartella che raccoglie i risultati delle migliori esecuzioni dei test (ogni cartella viene creata in automatico dallo script), una cartella con nome "data-ora" di esecuzione contiene:
<ul>
    <li> <em>myModel</em>: Salvataggio del modello di tensorflow.</li>
    <li> <em>infotest.txt</em>: Risultato dell'esecuzione del modello sul set di test contenete informazioni utili per capire la botà di esso. </li>
    <li><em>distribuzione_errori.jpg</em>: Grafico che mostra la distribuzione degli errori di decodifica commessi dal modello.</li>
    <li> log di tensorboard <em> train</em>, <em>validation</em>: Non presenti quelli dei risultati caricati, per problemi di spazio, ma in genere creati durante l'esecuzione.</li>
</ul>

Il resto dei file <em>.py</em> non sono stati utilizzati direttamente, ma forniti all'inizio del lavoro.
