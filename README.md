# Lucca Robot

Sistema prototipale di help-desk robotico per **Lucca Comics & Games**, sviluppato come proof of concept per modellare la **Competenza Culturale** come forma di ragionamento situato, abduttivo ed euristico.

Il progetto implementa un agente conversazionale che:
- interpreta indizi percettivi e input linguistici;
- aggiorna una distribuzione di credenza su possibili interessi, destinazioni o richieste del visitatore;
- usa una rete enciclopedica a grafo per combinare gerarchie semantiche e analogie culturali;
- seleziona azioni diverse a seconda dello stato epistemico corrente: domanda aperta, domanda mirata, suggerimento, risposta fattuale, proposta di alternative;
- incorpora una memoria di lavoro che tiene traccia del contesto locale della conversazione.

---

## Contesto teorico

Questo repository nasce nel quadro del lavoro:

**Saettone et al., _Heuristics of Culture: Situated and Dynamic Reasoning in Social Robots, ACT-R and Probabilistic Planning_**

L’idea di fondo è che la **Competenza Culturale (CC)** non vada intesa come una semplice banca dati di informazioni su gruppi o categorie sociali, ma come una **euristica di ragionamento** che consente a un agente di orientarsi in condizioni di conoscenza incompleta.

In questo senso, la cultura viene trattata come:

- una risorsa per formulare inferenze rapide ma rivedibili;
- una struttura enciclopedica di segni, pratiche, rimandi e pertinenze;
- una modalità di ragionamento situato che combina abduzione, analogia e aggiornamento continuo.

Il sistema non modella quindi la cultura come un modulo separato, ma come una proprietà emergente della coordinazione tra:

- percezione;
- memoria di lavoro;
- memoria dichiarativa enciclopedica;
- dinamiche di attivazione;
- aggiornamento bayesiano delle credenze;
- pianificazione dell’azione;
- revisione attraverso feedback.

---

## Idea architetturale

Il progetto combina componenti simboliche e subsimboliche.

### Componenti principali
- **Grafo enciclopedico**
  Organizza la conoscenza del dominio Lucca Comics & Games tramite relazioni gerarchiche e analogiche.
- **Observation model dinamico**
  Le osservazioni non sono interpretate tramite una tabella fissa, ma attraverso la struttura stessa del grafo.
- **Aggiornamento del belief**
  Il sistema mantiene una distribuzione di credenza sui “mondi possibili” e la aggiorna a ogni turno.
- **Working memory**
  Tiene traccia di ciò che è già stato suggerito, visitato, negato o discusso.
- **Planner**
  Seleziona l’atto dialogico più opportuno in base alla confidenza, al contesto e all’intento pragmatico.
- **Embeddings semantici**
  Usati per rafforzare o creare connessioni analogiche tra nodi del grafo.

---

## Struttura del repository

I file principali sono:

- `krobot_lucca.py`  
  Script principale del robot e del dialogo.
- `grafolucca.json`  
  Grafo enciclopedico del dominio.
- `.env`  
  File locale per la chiave OpenAI // che non c'è  su github.

A seconda della tua organizzazione del repository, puoi aggiungere anche:
- immagini;
- file di test;
- dump di output;
- documentazione del paper.

---

## Modello di conoscenza

Il sistema usa una rete semantica che rappresenta:

- **relazioni verticali** di tipo gerarchico (`appartenenza`);
- **relazioni orizzontali** di tipo analogico (`vicinanza`);
- **connessioni dinamiche** inferite tramite embedding (`vicinanza_dinamica`).

Questa organizzazione consente di distinguere tra:

- **generalità** e **specificità**;
- **comprensione** e **estensione**;
- **definizione gerarchica** e **rimando enciclopedico**.

Un nodo come `Berserk`, per esempio, può essere attivato:
- direttamente, come topic esplicito;
- verso l’alto, tramite `Manga` e `Fumetti`;
- lateralmente, tramite nodi analogicamente vicini come `Dylan Dog` o `Inuyasha`.

---

## Funzionamento del sistema

A ogni turno il robot segue una pipeline generale:

1. riceve un input che descrive ciò che il robot vede o ciò che il visitatore dice;
2. estrae osservazioni percettive e topic rilevanti;
3. aggiorna il belief sui mondi possibili;
4. propaga l’attivazione sul grafo in senso gerarchico e analogico;
5. tiene conto della memoria dialogica e dei feedback già ricevuti;
6. decide quale azione compiere;
7. genera una risposta in linguaggio naturale.

### Azioni possibili
Il planner può produrre azioni come:

- `chiedi_aperto`
- `chiedi_mirato`
- `suggerisci_stand`
- `suggerisci_evento`
- `suggerisci_adiacente`
- `mostra_connessione`
- `rispondi_fattuale`

---

## Spiegazione delle principali componenti

### 1. Percezione
La percezione è simulata a partire da input testuali. Il sistema prova a estrarre:

- un **percetto generico**, come `logo_indossato`, `costume_cosplay`, `busta_acquisti`;
- uno o più **topic**, come `topic:Berserk`, `topic:Concerto`, `topic:VR Arena ed eSports`;
- eventuali informazioni dialogiche, come:
  - `visited:...`
  - `bought:...`
  - `negated:...`

Quando disponibile, l’estrazione del topic viene supportata da un LLM. In caso di fallimento, entrano in gioco fallback simbolici basati su parole chiave.

### 2. Observation model dinamico
Il modello osservativo non è una semplice tabella esterna di corrispondenze.

Ogni osservazione viene ancorata a uno o più nodi del grafo e da lì l’attivazione si propaga in base a:
- tipo di relazione;
- peso dell’arco;
- distanza in hop;
- decadimento progressivo.

Questo consente di ottenere:
- distribuzioni diffuse per indizi vaghi;
- distribuzioni concentrate per indizi specifici.

### 3. Belief e mondi possibili
Il sistema mantiene una distribuzione di probabilità sui nodi rilevanti del dominio.

La sezione dei **mondi possibili** nell’output mostra, turno per turno:
- quali nodi sono considerati più plausibili;
- con quale peso;
- con quale grado di incertezza.

Questa distribuzione non coincide ancora con la decisione finale, ma costituisce il paesaggio epistemico su cui il planner lavora.

### 4. Working memory
La memoria di lavoro tiene traccia di:
- nodi già suggeriti;
- nodi già visitati;
- nodi parzialmente esplorati;
- referente dialogico;
- ultimo intento del robot;
- ultimo atto dell’utente;
- osservazioni cumulative.

Questo rende il dialogo cumulativo, contestuale e revisibile.

### 5. Pianificazione
Il planner osserva:
- il nodo più plausibile;
- la confidenza associata;
- la memoria di lavoro;
- l’intento dell’utente;
- l’eventuale priorità pragmatica di alcuni topic.

In base a questi elementi decide se:
- agire direttamente;
- chiedere;
- proporre un’alternativa;
- rispondere in modo fattuale;
- privilegiare un evento rispetto a un topic semanticamente più specifico.

### 6. Risposta linguistica
La risposta finale è generata solo dopo che il sistema ha già scelto:
- il target;
- l’azione;
- il contesto locale del grafo da mobilitare.

Questo permette di separare:
- **decisione cognitiva**
da
- **formulazione linguistica**

---

## Requisiti

Installa le dipendenze con:

```bash
pip install networkx sentence-transformers scikit-learn openai python-dotenv numpy
