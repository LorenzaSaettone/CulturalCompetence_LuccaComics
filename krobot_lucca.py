"""
lucca_robot.py
==============
Sistema robot help-desk Lucca Comics & Games.
Basato su: Saettone et al., "Heuristics of Culture: Situated and
Dynamic Reasoning in Social Robots, ACT-R and Probabilistic Planning"

INSTALLAZIONE:
    pip install networkx sentence-transformers scikit-learn openai python-dotenv

UTILIZZO:
    python lucca_robot.py

CHIAVE API:
    Crea un file .env nella stessa cartella con:
    OPENAI_API_KEY=sk-...

COMANDI DURANTE LA SESSIONE:
    scrivi liberamente quello che vedi o senti
    +     visitatore soddisfatto, se ne va
    -     visitatore insoddisfatto, ancora li'
    fine  chiudi il programma
"""

import json
import os
import re
import datetime
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

_embedding_model = None

def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("[embedding] Caricamento modello (~30s al primo avvio)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


SCRIPTS = {
    "Fumetti":                  "Padiglione A Fumetti, zona palazzetto dello sport.",
    "Videogiochi":              "Padiglione B e C Videogiochi, palazzetto dello sport.",
    "Film Serie Animazione":    "Cinema Piazza Anfiteatro e Loggiato del Pretorio.",
    "Eventi":                   "Palco Centrale e Sala Conferenze. Controlla gli orari!",
    "Servizi":                  "Bagni, ristoro e food truck vicino Piazza Santa Maria.",
    "Collezionabili Editoria":  "Piazza Santa Maria: stand collezionismo e giochi da tavolo.",
    "Marvel":       "Stand Marvel al Padiglione A. Fumetti, variant cover, firme autori.",
    "DC":           "Stand DC al Padiglione A. Fumetti e graphic novel.",
    "Bonelli":      "Stand Bonelli al Padiglione A. Dylan Dog, Tex e classici italiani.",
    "Manga":        "Zona manga al Padiglione A. Berserk, Naruto, One Piece.",
    "Disney Fumetti": "Stand Disney Fumetti al Padiglione A. Topolino e DuckTales.",
    "PlayStation":  "Area PlayStation al Padiglione B. GTA, The Last of Us, Detroit. "
                    "Conferenza GTA6 oggi ore 15:00 in Sala Conferenze 1!",
    "Nintendo":     "Area Nintendo al Padiglione B. Demo di Mario, Zelda, Pokemon.",
    "PC Indie":     "Area PC Games al Padiglione B. LucasArts, indie letterari, LoL.",
    "Cabinati Retro": "Zona Cabinati al Padiglione C. Street Fighter, Pac-Man, Donkey Kong.",
    "Cine-Marvel":          "Cinema Piazza Anfiteatro. Endgame, No Way Home, Doctor Strange.",
    "DC Film":              "Cinema Piazza Anfiteatro. Batman, Joker, Aquaman.",
    "Film Nerd Nostalgici": "Cinema Loggiato del Pretorio. Star Wars, LotR, Ghostbusters.",
    "Serie TV":             "Area Serie TV al Loggiato. Breaking Bad, Buffy, Stranger Things.",
    "Anime":                "Piazza del Collegio. Naruto, One Piece, Dan Da Dan, Inuyasha.",
    "Disney Film":          "Piazza del Collegio. Re Leone, Aladdin, Mulan.",
    "Concerto":         "Concerto Giorgio Vanni e Cristina D'Avena, Palco Centrale ore 21:00.",
    "Conferenza GTA6":  "Conferenza GTA6 in Sala Conferenze 1, ore 15:00-16:00.",
    "Sfida Cosplayer":  "Sfida Cosplayer sul Palco Centrale ore 16:30-18:00.",
    "VR Arena ed eSports": "Battle Arena al Padiglione C. VR e League of Legends.",
    "Funko Pop":    "Stand Funko Pop a Piazza Santa Maria.",
    "Lego":         "Stand Lego a Piazza Santa Maria. Set Star Wars, Marvel, Minecraft.",
    "Oggetti D&D":  "Stand D&D a Piazza Santa Maria. Manuali, dadi, miniature.",
    "Libri Fantasy": "Stand libri fantasy a Piazza Santa Maria.",
    "W.C.":   "Bagni vicino ai padiglioni principali.",
    "robot":  "Sono io! Il robot help-desk di Lucca Comics.",
    "Ristoro": "Food truck e bar vicino Piazza Santa Maria.",
    "Detroit Become Human": "Videogioco PlayStation sulla coscienza dei robot.",
    "Il Signore degli Anelli": "Film di Tolkien al Loggiato. Vicino a libri fantasy e D&D.",
    "Star Wars":        "Proiettato al Loggiato. Lego Star Wars allo stand collezionabili.",
    "Stranger Things":  "Serie TV al Loggiato. Fan spesso interessati agli oggetti D&D.",
    "Naruto_manga":     "Manga al Padiglione A. Versione anime a Piazza del Collegio.",
    "Naruto_anime":     "Anime a Piazza del Collegio. Manga al Padiglione A.",
    "OnePiece_manga":   "Manga One Piece al Padiglione A.",
    "OnePiece_anime":   "Anime One Piece a Piazza del Collegio.",
    "GTA":              "PlayStation al Padiglione B. Conferenza GTA6 ore 15:00!",
    "Batman":           "DC Comics al Padiglione A. Film Batman al Cinema Anfiteatro.",
    "Spider-Man":       "Marvel al Padiglione A. Film No Way Home al Cinema Anfiteatro.",
    "Dylan Dog":        "Bonelli al Padiglione A. Vicino a Buffy e Ghostbusters.",
    "Berserk":          "Manga al Padiglione A. Fan spesso apprezzano anche Dylan Dog.",
}

def get_script(node):
    return SCRIPTS.get(node, "")


INDIZI_TRANSAZIONALI = {
    "busta_acquisti",
}

TRANSAZIONALE_IMPLICA = {
    "busta_acquisti": None,
}

# ══════════════════════════════════════════════════════════════════════
# OBSERVATION MODEL — DINAMICO, DERIVATO DAL GRAFO
# ══════════════════════════════════════════════════════════════════════
#
# Principio (Eco): l'observation model non è una tabella separata ma
# emerge dalla struttura enciclopedica del grafo.
#   • Osservazione specifica → nodo-ancora profondo → distribuzione
#     concentrata (alta comprensione, bassa estensione)
#   • Osservazione generica  → nodi-ancora di tipo zona → distribuzione
#     diffusa (bassa comprensione, alta estensione)
#   • Nessun indizio         → uniforme su tutte le zone
#
# OBSERVATION_ANCHORS: mapping percetto → nodo/i nel grafo.
# È l'unico dato "manuale"; le probabilità vengono calcolate
# percorrendo il grafo (archi, pesi, embedding).
# ══════════════════════════════════════════════════════════════════════

OBSERVATION_ANCHORS = {
    # ── percetti visivi generici (nodo-ancora = zona) ────────────────
    # Il sensore rileva "logo su maglietta", non "fan DC".
    # Ancora generica → si distribuisce sulle zone pertinenti.
    "logo_indossato":   ["Fumetti", "Videogiochi", "Film Serie Animazione",
                         "Collezionabili Editoria"],
    "costume_cosplay":  ["Sfida Cosplayer", "Fumetti", "Anime",
                         "Film Serie Animazione"],
    "gadget_mano":      ["Collezionabili Editoria", "Funko Pop",
                         "Film Serie Animazione"],
    "busta_acquisti":   ["Collezionabili Editoria", "Fumetti", "Videogiochi"],
    "eta_bambino":      ["Nintendo", "Disney Film", "Anime"],
    "eta_adulto_nostalgico": ["Cabinati Retro", "Film Nerd Nostalgici",
                              "Bonelli", "Serie TV"],

    # ── percetti visivi specifici (nodo-ancora = istanza/contenuto) ──
    "logo_marvel":          ["Marvel"],
    "logo_dc":              ["DC"],
    "logo_nintendo":        ["Nintendo"],
    "logo_playstation":     ["PlayStation"],
    "logo_gta":             ["GTA"],
    "costume_supereroe":    ["Marvel", "DC", "Sfida Cosplayer"],
    "costume_anime":        ["Anime", "Manga", "Sfida Cosplayer"],
    "costume_jedi":         ["Star Wars", "Sfida Cosplayer"],
    "cappello_pirata":      ["OnePiece_anime", "OnePiece_manga"],
    "gadget_funko":         ["Funko Pop"],
    "zaino_nintendo":       ["Nintendo"],
    "maglietta_lotr":       ["Il Signore degli Anelli"],
    "maglietta_pokemon":    ["Nintendo"],     # Pokemon sotto Nintendo nel grafo
    "maglietta_stranger":   ["Stranger Things"],
    "borsa_lego":           ["Lego"],
    "busta_acquisti_dc":    ["DC"],
    "busta_acquisti_marvel": ["Marvel"],
    "busta_acquisti_anime": ["Anime", "Manga"],
    "busta_acquisti_nintendo": ["Nintendo"],
    "busta_acquisti_generica": ["Collezionabili Editoria"],

    # ── nessuna informazione ─────────────────────────────────────────
    "badge_generico":   [],      # → uniforme
    "nessun_indizio":   [],      # → uniforme
}

# Pesi per tipo di relazione nella propagazione graph-based
_RELATION_PROPAGATION_WEIGHT = {
    "appartenenza": 0.85,     # forte: Marvel → Fumetti
    "vicinanza":    0.50,     # media: Marvel → DC
    "vicinanza_dinamica": 0.35,  # debole: archi da embedding
    "necessità":    0.15,     # molto debole: Servizi → zone
}

# Cache per evitare di ricalcolare ad ogni turno per la stessa osservazione
_obs_model_cache = {}


def observation_model_from_graph(observation, G, hops=2):
    """
    Genera la distribuzione P(osservazione | mondo) percorrendo il grafo
    a partire dai nodi-ancora dell'osservazione.

    Meccanismo (ispirato ad Eco, modello enciclopedico):
    1. Dai nodi-ancora, propaga attivazione lungo gli archi del grafo
    2. Il peso di propagazione dipende dal tipo di relazione
    3. Ad ogni hop, l'attivazione decade
    4. Archi da embedding (L2) contribuiscono automaticamente

    Risultato: osservazioni specifiche → distribuzione concentrata
               osservazioni generiche → distribuzione diffusa
               nessun indizio        → distribuzione uniforme
    """
    # Cache check (invalidata ad ogni nuovo visitatore)
    cache_key = observation
    if cache_key in _obs_model_cache:
        return _obs_model_cache[cache_key]

    anchors = OBSERVATION_ANCHORS.get(observation, [])

    # Nessun ancora → distribuzione uniforme (massima estensione)
    if not anchors:
        return {}

    # Raccogli i nodi che partecipano al belief (zona, contenuto, istanza)
    belief_nodes = {
        n for n, d in G.nodes(data=True)
        if d.get("type") in ("zona", "contenuto", "istanza")
    }

    # Propagazione: spreading activation dai nodi-ancora
    activation = {}

    for anchor in anchors:
        if anchor not in G:
            continue

        # Il nodo-ancora stesso riceve attivazione piena
        activation[anchor] = activation.get(anchor, 0.0) + 1.0

        # Propaga per `hops` passi
        frontier = [(anchor, 1.0)]
        visited_in_walk = {anchor}

        for hop in range(hops):
            decay = 0.6 ** (hop + 1)   # decadimento per distanza
            next_frontier = []
            for node, parent_act in frontier:
                neighbors = set(G.successors(node)) | set(G.predecessors(node))
                for nb in neighbors:
                    if nb in visited_in_walk:
                        continue
                    # Peso dell'arco
                    if G.has_edge(node, nb):
                        ed = G[node][nb]
                    else:
                        ed = G[nb][node]
                    relation = ed.get("relation", "altro")
                    rel_weight = _RELATION_PROPAGATION_WEIGHT.get(relation, 0.20)
                    edge_weight = ed.get("weight", 0.4)

                    # Attivazione = decadimento × peso relazione × peso arco
                    act = parent_act * decay * rel_weight * edge_weight

                    if act > 0.005:   # soglia minima
                        old = activation.get(nb, 0.0)
                        activation[nb] = max(old, act)  # max, non somma (evita inflazione)
                        next_frontier.append((nb, act))
                        visited_in_walk.add(nb)

            frontier = next_frontier

    # Filtra solo i nodi che partecipano al belief
    lk_map = {n: v for n, v in activation.items() if n in belief_nodes}

    # Normalizza come likelihood (il bayesian_update farà il resto)
    if lk_map:
        max_val = max(lk_map.values())
        if max_val > 0:
            lk_map = {n: round(v / max_val, 4) for n, v in lk_map.items()}

    _obs_model_cache[cache_key] = lk_map
    return lk_map


def invalidate_obs_cache():
    """Invalida la cache dell'observation model (chiamare ad ogni nuovo visitatore)."""
    _obs_model_cache.clear()


def get_valid_observations():
    """Restituisce la lista di osservazioni riconosciute dal sistema."""
    return list(OBSERVATION_ANCHORS.keys())


def is_known_observation(obs):
    """Controlla se un'osservazione è nota al sistema."""
    return obs in OBSERVATION_ANCHORS


# Alias di compatibilità: il vecchio OBSERVATION_MODEL viene calcolato on-demand.
# NOTA: questo dict viene popolato dopo il caricamento del grafo (vedi init).
OBSERVATION_MODEL = {}


def _default_weight(relation):
    return {
        "appartenenza": 1.0,
        "vicinanza": 0.65,
        "necessita": 0.20,
        "necessità": 0.20,
    }.get(relation, 0.4)


def load_graph(path="grafolucca.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], type=node["type"], embedding=None)
    for edge in data["edges"]:
        G.add_edge(edge["from"], edge["to"],
                   relation=edge.get("relation", "altro"),
                   weight=_default_weight(edge.get("relation", "altro")),
                   sim_score=None, source="json",
                   frame=edge.get("frame", None),
                   pertinenza=edge.get("pertinenza", []),
                   narrativa=edge.get("narrativa", None))
    return G


def compute_node_embeddings(G):
    model = _get_model()
    nodes = list(G.nodes())
    texts = [f"{n} ({G.nodes[n].get('type','')}). {get_script(n)[:100]}" for n in nodes]
    embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for node, emb in zip(nodes, embs):
        G.nodes[node]["embedding"] = emb
    print(f"[L1-embedding] {len(nodes)} vettori calcolati.")
    return G


def update_analogical_edges(G, sim_threshold=0.52, max_new_edges=12,
                             reinforce_existing=True):
    nodes_with_emb = [n for n in G.nodes() if G.nodes[n].get("embedding") is not None]
    embs       = np.array([G.nodes[n]["embedding"] for n in nodes_with_emb])
    sim_matrix = cosine_similarity(embs)
    new_edges  = []
    updated    = 0
    for i, n1 in enumerate(nodes_with_emb):
        for j, n2 in enumerate(nodes_with_emb):
            if i >= j:
                continue
            sim = float(sim_matrix[i, j])
            if reinforce_existing:
                for src, dst in [(n1, n2), (n2, n1)]:
                    if G.has_edge(src, dst) and G[src][dst].get("relation") == "vicinanza":
                        old_w = G[src][dst]["weight"]
                        G[src][dst]["weight"] = round(0.6 * old_w + 0.4 * sim, 3)
                        G[src][dst]["sim_score"] = round(sim, 3)
                        updated += 1

            # NON elif
            if sim >= sim_threshold and not G.has_edge(n1, n2) and not G.has_edge(n2, n1):
                new_edges.append((n1, n2, round(sim, 3)))


    new_edges.sort(key=lambda x: x[2], reverse=True)
    added = 0
    for n1, n2, sim in new_edges[:max_new_edges]:
        G.add_edge(n1, n2, relation="vicinanza_dinamica",
           weight=sim, sim_score=sim, source="embedding")

        G.add_edge(n2, n1, relation="vicinanza_dinamica",
                weight=sim, sim_score=sim, source="embedding")
        added += 2
    print(f"[L2-analogia] Aggiornati: {updated}, nuovi dinamici: {added}")
    return G


def initialize_system(path="grafolucca.json"):
    print("\n=== Inizializzazione sistema robot Lucca Comics ===")
    G = load_graph(path)
    print(f"Grafo base: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    G = compute_node_embeddings(G)
    G = update_analogical_edges(G)
    print(f"Grafo arricchito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    print("=== Sistema pronto ===\n")
    return G


def init_working_memory():
    return {
        "suggeriti": set(),
        "confermati": set(),
        "turno": 0,
        "turni_senza_info": 0,
        "parzialmente_esplorati": set(),

        # memoria dialogica
        "referente_dialogico": None,
        "ultimo_intento": None,
        "ultimo_atto_utente": None,

        # osservazioni cumulative per pertinenza situata
        "_obs_cumulative": [],
    }


def prior_contestuale(G, ora_simulata=None):
    ora    = ora_simulata if ora_simulata is not None else datetime.datetime.now().hour
    belief = init_belief(G)
    if 12 <= ora < 15:
        boost  = {"Servizi": 0.20, "Ristoro": 0.25}
        motivo = f"ora di pranzo ({ora}:xx) — Ristoro in evidenza"
    elif 20 <= ora < 23:
        boost  = {"Concerto": 0.35, "Eventi": 0.20}
        motivo = f"fascia serale ({ora}:xx) — Concerto in evidenza"
    elif 16 <= ora < 19:
        boost  = {"Sfida Cosplayer": 0.25, "Eventi": 0.15}
        motivo = f"fascia pomeridiana ({ora}:xx) — Sfida Cosplayer in evidenza"
    elif 14 <= ora < 16:
        boost  = {"Conferenza GTA6": 0.30, "Videogiochi": 0.15}
        motivo = f"fascia conferenza ({ora}:xx) — GTA6 in evidenza"
    else:
        return belief
    for nodo, valore in boost.items():
        if nodo in belief:
            belief[nodo] = valore
    total  = sum(belief.values())
    belief = {n: v / total for n, v in belief.items()}
    print(f"  [prior contestuale] {motivo}")
    return belief


def init_belief(G):
    relevant = [
        n for n, d in G.nodes(data=True)
        if d["type"] in ("zona", "contenuto", "istanza")
    ]
    p = 1.0 / len(relevant)
    return {n: p for n in relevant}


def bayesian_update(belief, observation, G=None):
    """
    Aggiornamento bayesiano con observation model derivato dal grafo.
    Se G è fornito, calcola le likelihood dinamicamente dalla struttura
    enciclopedica; altrimenti fallback al dict statico di compatibilità.
    """
    epsilon = 0.005
    if G is not None:
        lk_map = observation_model_from_graph(observation, G)
    else:
        lk_map = OBSERVATION_MODEL.get(observation, {})
    unnorm  = {n: lk_map.get(n, epsilon) * p for n, p in belief.items()}
    total   = sum(unnorm.values())
    if total == 0:
        return belief
    return {n: v / total for n, v in unnorm.items()}


def top_beliefs(belief, k=5):
    return sorted(belief.items(), key=lambda x: x[1], reverse=True)[:k]


def confidenza_massima(belief):
    tops = top_beliefs(belief, 1)
    return tops[0] if tops else (None, 0.0)


def navigate(G, start_node, hops=2, min_score=0.05, osservazioni=None):
    """Spreading activation con boost pertinenziale situato."""
    if start_node not in G:
        return []

    obs_set = set(osservazioni) if osservazioni else set()

    results = {}
    frontier = [(start_node, 1.0, "start")]
    best_score = {start_node: 1.0}

    for _ in range(hops):
        next_frontier = []
        for node, score, _ in frontier:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            for nb in neighbors:
                ed = G[node][nb] if G.has_edge(node, nb) else G[nb][node]
                base_weight = ed.get("weight", 0.4)

                # boost pertinenziale situato
                pert = ed.get("pertinenza", [])
                if obs_set and pert and obs_set.intersection(pert):
                    base_weight = min(1.0, base_weight * 1.4)

                new_score = score * base_weight
                rel = ed.get("relation", "altro")

                if new_score < min_score:
                    continue

                if new_score > best_score.get(nb, 0):
                    best_score[nb] = new_score
                    results[nb] = (new_score, rel)
                    next_frontier.append((nb, new_score, rel))

        frontier = next_frontier

    results.pop(start_node, None)
    return sorted(
        [(n, s, r) for n, (s, r) in results.items()],
        key=lambda x: x[1],
        reverse=True
    )


SOGLIA_AGISCI = 0.30
SOGLIA_CHIEDI = 0.10

def seleziona_topic_pragmatico(raw, topic_nodes, G, wm=None):
    if not topic_nodes:
        return None

    text = raw.lower()

    event_cues = [
        "concerto", "conferenza", "evento", "spettacolo",
        "sentire", "ascoltare", "vedere dal vivo", "palco",
        "giorgio vanni", "cristina d'avena", "cristina d'avena"
    ]
    location_cues = [
        "dove", "dov'è", "dov'e", "si trova", "c'è", "c'e",
        "esiste", "padiglione", "stand", "zona"
    ]
    interest_cues = [
        "cerco", "mi interessa", "mi piace", "vorrei",
        "sono qui per", "voglio", "mi piacciono"
    ]
    alternative_cues = [
        "altro", "altri suggerimenti", "cos'altro", "e poi", "e dopo"
    ]

    def node_score(node):
        score = 0.0
        script = get_script(node).lower() if node in G.nodes() else ""
        node_type = G.nodes[node].get("type", "") if node in G.nodes() else ""

        # 1. compatibilita' con frame evento
        if any(c in text for c in event_cues):
            if any(k in script for k in ["ore ", "palco", "concerto", "conferenza"]):
                score += 6.0
            if node in {"Concerto", "Conferenza GTA6", "Sfida Cosplayer"}:
                score += 4.0

        # 2. compatibilita' con frame fattuale / localizzazione
        if any(c in text for c in location_cues):
            score += 2.0
            if node_type in ("zona", "contenuto", "istanza"):
                score += 1.0

        # 3. compatibilita' con richiesta di interesse generico
        if any(c in text for c in interest_cues):
            if node_type in ("contenuto", "istanza", "zona"):
                score += 1.5

        # 4. se l'utente chiede alternative, penalizza il gia' visto
        if any(c in text for c in alternative_cues):
            if wm and node in wm.get("confermati", set()):
                score -= 5.0

        # 5. bonus se il nodo e' menzionato esplicitamente nel testo
        surface = node.lower().replace("_", " ")
        if surface in text:
            score += 3.0

        return score

    ranked = sorted(topic_nodes, key=node_score, reverse=True)
    return ranked[0]


def plan(belief, G, wm=None):
    best_node, best_score = confidenza_massima(belief)

    if wm:
        obs = wm.get("_obs_cumulative", [])
        has_semantic_signal = any(
            isinstance(o, str) and o.startswith(("topic:", "visited:", "bought:"))
            for o in obs
        )
        if "logo_indossato" in obs and not has_semantic_signal:
            return "chiedi_aperto", None, min(best_score, SOGLIA_CHIEDI)

    if best_score >= SOGLIA_AGISCI:
        if wm and best_node in wm["confermati"]:
            alt = find_affine_alternative(G, best_node, wm, max_hops=2)
            if alt:
                return "suggerisci_adiacente", alt[0], best_score * 0.85

        if wm and best_node in wm.get("parzialmente_esplorati", set()):
            fratelli = []
            for child in G.successors(best_node):
                if G[best_node][child].get("relation") == "appartenenza":
                    if child not in wm["confermati"] and child not in wm["suggeriti"] and get_script(child):
                        fratelli.append((child, G[best_node][child].get("weight", 1.0), "appartenenza"))

            fratelli.sort(key=lambda x: x[1], reverse=True)

            if fratelli:
                return "suggerisci_stand", fratelli[0][0], best_score * 0.90
            alt = find_affine_alternative(G, best_node, wm, max_hops=2)
            if alt:
                return "suggerisci_adiacente", alt[0], best_score * 0.80

        script = get_script(best_node)
        if any(kw in script for kw in ["ore ", "15:00", "16:30", "21:00"]):
            return "suggerisci_evento", best_node, best_score

        # Connessione analogica se il nodo è già noto al visitatore:
        #   - già suggerito dal robot in un turno precedente, OPPURE
        #   - già visitato (confermato), OPPURE
        #   - il suo genitore (zona/contenuto) è già visitato
        #     (es. Berserk sotto Manga sotto Fumetti: se Fumetti è
        #      confermato, il visitatore già conosce quell'area).
        # Principio enciclopedico: comprensione prima, estensione poi —
        # ma se la comprensione è già acquisita, si passa all'estensione.
        gia_noto = False
        if wm is not None:
            gia_noto = (best_node in wm.get("suggeriti", set())
                        or best_node in wm.get("confermati", set()))
            if not gia_noto:
                # Controlla se un antenato (via appartenenza) è confermato
                for parent in G.predecessors(best_node):
                    if (G.has_edge(parent, best_node)
                            and G[parent][best_node].get("relation") == "appartenenza"
                            and parent in wm.get("confermati", set())):
                        gia_noto = True
                        break
                if not gia_noto:
                    # Secondo livello: nonno (zona → contenuto → istanza)
                    for parent in G.predecessors(best_node):
                        if not (G.has_edge(parent, best_node)
                                and G[parent][best_node].get("relation") == "appartenenza"):
                            continue
                        for grandparent in G.predecessors(parent):
                            if (G.has_edge(grandparent, parent)
                                    and G[grandparent][parent].get("relation") == "appartenenza"
                                    and grandparent in wm.get("confermati", set())):
                                gia_noto = True
                                break
                        if gia_noto:
                            break

        nav    = navigate(G, best_node, hops=1)
        vicini = [(n, s, r) for n, s, r in nav
                  if "vicinanza" in r and get_script(n)
                  and (wm is None or n not in wm["confermati"])]
        if vicini and gia_noto:
            return "mostra_connessione", best_node, best_score
        return "suggerisci_stand", best_node, best_score

    elif best_score >= SOGLIA_CHIEDI:
        if wm and best_node in wm["confermati"]:
            alt = find_affine_alternative(G, best_node, wm, max_hops=2)
            if alt:
                return "suggerisci_adiacente", alt[0], best_score * 0.85
            return "chiedi_aperto", None, best_score
        if wm and best_node in wm.get("parzialmente_esplorati", set()):
            fratelli = []
            for child in G.successors(best_node):
                if G[best_node][child].get("relation") == "appartenenza":
                    if child not in wm["confermati"] and child not in wm["suggeriti"] and get_script(child):
                        fratelli.append((child, G[best_node][child].get("weight", 1.0), "appartenenza"))

            fratelli.sort(key=lambda x: x[1], reverse=True)

            if fratelli:
                return "chiedi_mirato", fratelli[0][0], best_score * 0.90
            alt = find_affine_alternative(G, best_node, wm, max_hops=2)
            if alt:
                return "suggerisci_adiacente", alt[0], best_score * 0.80
        return "chiedi_mirato", best_node, best_score

    if wm and wm.get("turni_senza_info", 0) >= 2:
        return "offri_mappa", None, best_score

    return "chiedi_aperto", None, best_score


def stampa_stato_epistemico(belief, G, azione, target, confidenza, osservazioni, wm):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  STATO EPISTEMICO — turno {wm['turno']}")
    print(sep)

    print("\n  MONDI POSSIBILI (distribuzione di credenza):")
    for nodo, prob in top_beliefs(belief, k=6):
        barre = "#" * int(prob * 40)
        visited = " [gia' visitato]" if wm and nodo in wm["confermati"] else ""
        print(f"    {nodo:<30} {prob:5.1%}  {barre}{visited}")

    print(f"\n  Mondo attuale piu' probabile : '{target or 'incerto'}'")
    print(f"  Confidenza                   : {confidenza:.1%}  ", end="")
    if confidenza >= SOGLIA_AGISCI:
        print("-> sufficiente per agire")
    elif confidenza >= SOGLIA_CHIEDI:
        print("-> sufficiente per chiedere in modo mirato")
    else:
        print("-> troppo bassa, serve piu' informazione")

    if target:
        print(f"\n  ENCICLOPEDIA LOCALE ELICITATA (da '{target}'):")

        # parent gerarchici
        parents = []
        for pred in G.predecessors(target):
            if G[pred][target].get("relation") == "appartenenza":
                parents.append((pred, G[pred][target].get("weight", 1.0)))
        parents.sort(key=lambda x: x[1], reverse=True)

        # figli gerarchici
        children = []
        for succ in G.successors(target):
            if G[target][succ].get("relation") == "appartenenza":
                children.append((succ, G[target][succ].get("weight", 1.0)))
        children.sort(key=lambda x: x[1], reverse=True)

        # analogie
        nav = navigate(G, target, hops=1)
        analogie = [(n, s) for n, s, r in nav if "vicinanza" in r]
        analogie.sort(key=lambda x: x[1], reverse=True)

        if parents:
            print("    Gerarchia sopra (categorie):")
            for n, s in parents[:5]:
                tag = " [gia' visitato]" if wm and n in wm["confermati"] else ""
                print(f"      {n} (score: {s:.2f}){tag}")

        if children:
            print("    Gerarchia sotto (contenuti):")
            for n, s in children[:5]:
                tag = " [gia' visitato]" if wm and n in wm["confermati"] else ""
                print(f"      {n} (score: {s:.2f}){tag}")

        if analogie:
            print("    Analogie orizzontali:")
            for n, s in analogie[:5]:
                tag = " [gia' visitato]" if wm and n in wm["confermati"] else ""
                print(f"      {n} (score: {s:.2f}){tag}")

    if wm and wm["confermati"]:
        print(f"\n  Working memory — gia' visitati: {sorted(wm['confermati'])}")

    if wm and wm.get("referente_dialogico"):
        print(f"  Referente dialogico           : {wm['referente_dialogico']}")

    if wm and wm.get("ultimo_intento"):
        print(f"  Ultimo intento robot          : {wm['ultimo_intento']}")

    if wm and wm.get("ultimo_atto_utente"):
        print(f"  Ultimo atto utente            : {wm['ultimo_atto_utente']}")

    print(f"\n  AZIONE PIANIFICATA : {azione}")
    print(f"  Osservazioni usate : {osservazioni}")
    print(sep)

def _openai_call(system, user, temperature=0.7, max_tokens=200):
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        resp   = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Errore: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  GRAPH-BASED VISIT/PURCHASE HELPERS
# ══════════════════════════════════════════════════════════════════════

PURCHASE_VERBS = [
    "comprato", "comprata", "comprati", "comprate",
    "preso", "presa", "presi", "prese",
    "acquistato", "acquistata", "acquistati", "acquistate",
    "compro", "prendo", "acquisto",
]

VISIT_VERBS = [
    "gia' stato", "gia stato", "già stato",
    "gia' stata", "gia stata", "già stata",
    "ci sono stato", "ci sono stata",
    "sono passato", "sono passata",
    "ho visitato", "ho visto", "ho gia' visto", "ho già visto",
    "l'ho visto", "l'ho visitato",
    "gia' visitato", "già visitato", "gia visitato",
    "vengo da", "vengo dallo", "vengo dalla", "vengo dal",
    "ero a", "ero allo", "ero alla", "ero al",
    "tornato da", "tornata da", "torno da", "torno dal",
]

_node_aliases = {}

def _build_node_aliases(G):
    aliases = {}
    for nid in G.nodes():
        aliases[nid.lower()] = nid
        aliases[nid.lower().replace("_", " ")] = nid
    extra = {
        "funko": "Funko Pop", "funko pop": "Funko Pop",
        "lego": "Lego", "d&d": "Oggetti D&D", "dnd": "Oggetti D&D",
        "dungeon and dragons": "Oggetti D&D", "dungeons and dragons": "Oggetti D&D",
        "libri fantasy": "Libri Fantasy", "fantasy": "Libri Fantasy",
        "marvel": "Marvel", "dc": "DC",
        "bonelli": "Bonelli", "manga": "Manga", "anime": "Anime",
        "disney": "Disney Fumetti", "disney film": "Disney Film",
        "playstation": "PlayStation", "nintendo": "Nintendo",
        "cabinati": "Cabinati Retro", "retro": "Cabinati Retro",
        "pc indie": "PC Indie", "videogiochi": "Videogiochi",
        "videogame": "Videogiochi", "fumetti": "Fumetti",
        "collezionabili": "Collezionabili Editoria",
        "serie tv": "Serie TV", "serie": "Serie TV",
        "film": "Film Serie Animazione", "cinema": "Film Serie Animazione",
        "eventi": "Eventi", "concerto": "Concerto",
        "cosplay": "Sfida Cosplayer",
        "gta": "GTA", "star wars": "Star Wars",
        "signore degli anelli": "Il Signore degli Anelli",
        "lotr": "Il Signore degli Anelli",
        "stranger things": "Stranger Things",
        "batman": "Batman", "spider-man": "Spider-Man",
        "dylan dog": "Dylan Dog", "berserk": "Berserk",
        "naruto": "Naruto_manga", "one piece": "OnePiece_manga",
        "pokemon": "Pokemon", "zelda": "Zelda", "mario": "Mario",
        "minecraft": "Minecraft", "ghostbusters": "Ghostbusters",
        "breaking bad": "Breaking Bad", "buffy": "Buffy",
        "squid game": "Squid Game", "inuyasha": "Inuyasha",
        "ranma": "Ranma", "dan da dan": "Dan Da Dan",
    }
    for k, v in extra.items():
        if v in G.nodes():
            aliases[k] = v
    return aliases


def _match_text_to_nodes(text, G):
    global _node_aliases
    if not _node_aliases:
        _node_aliases = _build_node_aliases(G)
    text_lower = text.lower()
    found = []
    for alias in sorted(_node_aliases.keys(), key=len, reverse=True):
        if alias in text_lower:
            nid = _node_aliases[alias]
            if nid not in found:
                found.append(nid)
    return found

SEMANTIC_FEATURE_KEYWORDS = {
    "mood:dark": ["dark", "oscuro", "oscura", "cupe", "cupo", "gotico", "gotica", "nero", "nera"],
    "genre:horror": ["horror", "orrore", "mostri", "demoni", "incubo", "paranormale", "vampiri", "pauroso"],
    "theme:fantasy": ["fantasy", "epico", "epica", "magia", "spade", "demoni", "draghi", "medioevale"],
    "tone:nostalgia": ["nostalg", "anni 80", "anni ottanta", "retro", "classici", "cult"],
    "tone:comedy": ["comico", "comica", "divertente", "umorismo", "ironico", "ironica"],
}


def extract_semantic_features(text):
    text_lower = text.lower()
    found = []
    for feat, keywords in SEMANTIC_FEATURE_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            found.append(feat)
    return found


def get_descendants_by_appartenenza(G, node_id, max_depth=3):
    results = []
    frontier = [(node_id, 0)]
    seen = {node_id}
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for child in G.successors(current):
            if G[current][child].get("relation") != "appartenenza":
                continue
            if child in seen:
                continue
            seen.add(child)
            results.append((child, depth + 1))
            frontier.append((child, depth + 1))
    return results


def _semantic_match_score(text, keywords):
    score = 0.0
    for kw in keywords:
        if kw and kw in text:
            score += 1.0
    return score


def rank_specific_candidates(G, base_topic, raw_text, wm=None, max_results=3):
    if base_topic not in G:
        return []

    raw_lower = raw_text.lower()
    obs = set(wm.get("_obs_cumulative", [])) if wm else set()
    features = set(extract_semantic_features(raw_text))

    # Raffina solo se c'e' un segnale aggiuntivo reale: feature semantica
    # o menzione esplicita di un discendente del topic.
    descendants = get_descendants_by_appartenenza(G, base_topic, max_depth=3)
    if not descendants:
        return []

    explicit_nodes = set(_match_text_to_nodes(raw_text, G))
    has_extra_signal = bool(features) or any(node in explicit_nodes for node, _ in descendants)
    if not has_extra_signal:
        return []

    candidates = []
    for node, depth in descendants:
        node_type = G.nodes[node].get("type")
        if node_type not in ("contenuto", "istanza"):
            continue

        score = 0.0
        searchable = " ".join([node.lower().replace("_", " "), get_script(node).lower()])

        if node in explicit_nodes:
            score += 5.0
        score += _semantic_match_score(raw_lower, [node.lower(), node.lower().replace("_", " ")]) * 4.0

        # bonus alle istanze: se l'utente specifica una qualita', e' utile scendere di livello
        if node_type == "istanza":
            score += 0.4

        # il livello piu' vicino al topic pesa di piu', ma senza impedire alle istanze forti di emergere
        score += max(0.0, 1.3 - 0.25 * depth)

        # cerca segnali in frame/narrative/pertinenza sugli archi incidenti
        incident_texts = []
        for src, dst in list(G.in_edges(node)) + list(G.out_edges(node)):
            ed = G[src][dst]
            frame = ed.get("frame") or ""
            narrativa = ed.get("narrativa") or ""
            pert = " ".join(ed.get("pertinenza", []))
            incident_texts.append(f"{frame} {narrativa} {pert}".lower())

        incident_blob = " ".join(incident_texts)
        combined = f"{searchable} {incident_blob}"

        feature_keywords = []
        for feat in features:
            feature_keywords.extend(SEMANTIC_FEATURE_KEYWORDS.get(feat, []))
        score += 0.9 * _semantic_match_score(combined, feature_keywords)

        # qualche euristica culturale minima, utile se il grafo non esplicita abbastanza il frame
        if "mood:dark" in features or "genre:horror" in features:
            if node in {"Berserk", "Dylan Dog", "Batman", "Joker"}:
                score += 3.0
            if any(k in combined for k in ["dark", "horror", "orrore", "oscur", "incubo", "paranormale"]):
                score += 2.0

        if "theme:fantasy" in features:
            if node in {"Il Signore degli Anelli", "Zelda", "Libri Fantasy", "Oggetti D&D", "Berserk"}:
                score += 2.5

        if "tone:nostalgia" in features:
            if node in {"Ghostbusters", "Ritorno al Futuro", "Goonies", "Pac-Man", "Donkey Kong"}:
                score += 2.5

        if score > 0:
            candidates.append((node, round(score, 3), depth, node_type))

    candidates.sort(key=lambda x: (x[1], x[3] == "istanza", -x[2]), reverse=True)

    seen = set()
    out = []
    for node, score, depth, node_type in candidates:
        if node in seen:
            continue
        seen.add(node)
        out.append((node, score, depth, node_type))
        if len(out) >= max_results:
            break
    return out


def enrich_topics_with_instances(raw_text, detected_obs, G, wm=None):
    detected = list(detected_obs)
    added = []
    base_topics = [o.split(":", 1)[1] for o in detected if o.startswith("topic:")]

    for topic_node in base_topics:
        ranked = rank_specific_candidates(G, topic_node, raw_text, wm=wm, max_results=3)
        for node, score, depth, node_type in ranked:
            tag = f"topic:{node}"
            if tag not in detected and tag not in added:
                added.append(tag)
    return detected + added, added


def get_ancestors_by_appartenenza(G, node_id):
    ancestors = []
    for pred in G.predecessors(node_id):
        if G.has_edge(pred, node_id):
            if G[pred][node_id].get("relation", "") == "appartenenza":
                ancestors.append(pred)
                ancestors.extend(get_ancestors_by_appartenenza(G, pred))
    return ancestors


def mark_visited(G, node_id, wm, source="generic"):
    wm["confermati"].add(node_id)
    print(f"  [wm] '{node_id}' marcato come gia' visitato ({source})")
    for anc in get_ancestors_by_appartenenza(G, node_id):
        wm["parzialmente_esplorati"].add(anc)
        print(f"  [wm] '{anc}' marcato come parzialmente esplorato")


def find_affine_alternative(G, node_id, wm, max_hops=2):
    for hops in range(1, max_hops + 1):
        nav = navigate(G, node_id, hops=hops)
        candidates = [(n, s, r) for n, s, r in nav
                       if "vicinanza" in r
                       and n not in wm["confermati"]
                       and n not in wm["suggeriti"]
                       and get_script(n)]
        if candidates:
            return candidates[0]
    nav = navigate(G, node_id, hops=max_hops)
    candidates = [(n, s, r) for n, s, r in nav
                   if n not in wm["confermati"]
                   and n not in wm["suggeriti"]
                   and get_script(n)]
    return candidates[0] if candidates else None

def boost_topic_with_graph(belief, G, nodo_topic):
    if nodo_topic not in G:
        return belief

    new_belief = {n: v * 0.15 for n, v in belief.items()}

    # nodo esplicito
    if nodo_topic in new_belief:
        new_belief[nodo_topic] += 0.55

    # parent / gerarchia
    for pred in G.predecessors(nodo_topic):
        if G[pred][nodo_topic].get("relation") == "appartenenza":
            if pred in new_belief:
                new_belief[pred] += 0.20

            for pred2 in G.predecessors(pred):
                if G[pred2][pred].get("relation") == "appartenenza":
                    if pred2 in new_belief:
                        new_belief[pred2] += 0.10

    # vicinanze semantiche
    nav = navigate(G, nodo_topic, hops=1, min_score=0.01)
    for nb, s, r in nav:
        if nb in new_belief and "vicinanza" in r:
            new_belief[nb] += 0.10 * s

    total = sum(new_belief.values())
    if total > 0:
        new_belief = {n: v / total for n, v in new_belief.items()}

    return new_belief

def percezione_da_testo(descrizione: str, G=None) -> list:
    # Costruisci lista nodi dal grafo
    if G is not None:
        nodi_grafo = sorted([n for n in G.nodes()
                             if n not in ("W.C.", "robot", "Ristoro")])
    else:
        nodi_grafo = [
            "Fumetti", "Videogiochi", "Film Serie Animazione", "Eventi",
            "Collezionabili Editoria", "Marvel", "DC", "Bonelli", "Manga",
            "Disney Fumetti", "PlayStation", "Nintendo", "PC Indie",
            "Cabinati Retro", "Cine-Marvel", "DC Film", "Film Nerd Nostalgici",
            "Serie TV", "Anime", "Disney Film", "Concerto", "Conferenza GTA6",
            "Sfida Cosplayer", "VR Arena ed eSports", "Funko Pop", "Lego",
            "Oggetti D&D", "Libri Fantasy",
        ]
    osservazioni_valide = get_valid_observations()

    system = "Sei il modulo di percezione di un robot help-desk a Lucca Comics."
    user   = f"""Ricevi l'input di un operatore che descrive un visitatore o riporta cosa dice.
Devi produrre DUE LIVELLI di classificazione:

LIVELLO 1 — PERCETTO (cosa vedo fisicamente):
Scegli UNA etichetta dalla lista: {osservazioni_valide}
- "logo_indossato": il visitatore indossa qualcosa con un logo/brand/personaggio
- "costume_cosplay": il visitatore e' travestito/in costume
- "gadget_mano": il visitatore tiene in mano un GADGET/PUPAZZO/OGGETTO della fiera (Funko Pop, spada giocattolo, dado D&D, ecc.)
  NON cibo, bevande, telefoni o oggetti quotidiani — quelli NON sono indizi culturali
- "busta_acquisti": il visitatore porta una BUSTA DI PLASTICA con acquisti fatti in fiera
  (NON una borsa/zaino con un logo — quello e' "logo_indossato")
- "eta_bambino": il visitatore e' un bambino piccolo
- "badge_generico": nessun indizio visivo particolare
- "nessun_indizio": l'input non contiene info visive

LIVELLO 2 — INTERPRETAZIONE (cosa deduco dal contenuto):
Scegli uno o piu' tag dal tipo appropriato:

A) INTERESSE DEDOTTO — il logo/costume/gadget suggerisce un interesse
   Restituisci "topic:NomeNodo" dalla lista nodi
   Es: maglietta Batman → "topic:DC" e "topic:Batman"
   Es: costume da Naruto → "topic:Anime" e "topic:Naruto_anime"
   Es: maglietta Ghostbusters → "topic:Ghostbusters"
   Es: cappellino SEGA → "topic:Cabinati Retro"

B) INTERESSE DICHIARATO — il visitatore dice cosa cerca
   Restituisci "topic:NomeNodo"
   Es: "cerco videogiochi" → "topic:Videogiochi"
   Es: "mi piace Star Wars" → "topic:Star Wars"

C) GIA' VISITATO — il visitatore dice di essere gia' stato
   Restituisci "visited:NomeNodo"

D) GIA' ACQUISTATO — il visitatore dice di aver comprato
   Restituisci "bought:NomeNodo"

Lista nodi validi: {nodi_grafo}

REGOLE IMPORTANTI:
- Restituisci SEMPRE un percetto + una o piu' interpretazioni
- Per loghi/magliette/gadget: percetto ("logo_indossato") + topic del brand
- Per costumi: percetto ("costume_cosplay") + topic del personaggio/universo
- Per buste: percetto ("busta_acquisti") + bought del contenuto se deducibile
- Se il visitatore PARLA (dice cosa cerca, dove e' stato): usa solo interpretazioni
- Se il brand/personaggio NON e' nella lista nodi, mappa al nodo piu' vicino SOLO se il contenuto e' comunque riconoscibile
  Es: "maglietta Demon Slayer" → "topic:Anime" e "topic:Manga"
  Es: "maglietta Halo" → "topic:Videogiochi"
- Se il logo/disegno NON e' riconosciuto o l'operatore dice esplicitamente "logo non riconosciuto", "logo non identificato", "non so che logo sia": NON inventare topic generici.
  In quel caso restituisci solo il percetto visivo generico, di solito "logo_indossato".
- Se non riesci a dedurre il contenuto specifico: solo il percetto generico
- Oggetti NON culturali (cibo, panini, bevande, telefoni, borse generiche senza logo, ombrelli, ecc.) NON sono indizi. Restituisci "nessun_indizio".

Esempi:
"maglietta Batman" → ["logo_indossato", "topic:DC", "topic:Batman"]
"costume da Naruto" → ["costume_cosplay", "topic:Anime", "topic:Naruto_anime"]
"persona con un panino in mano" → ["nessun_indizio"]
"persona che mangia" → ["nessun_indizio"]
"persona con bottiglia d'acqua" → ["nessun_indizio"]
"maglietta Ghostbusters" → ["logo_indossato", "topic:Ghostbusters", "topic:Film Nerd Nostalgici"]
"busta con roba DC" → ["busta_acquisti", "bought:DC"]
"borsa con logo Batman" → ["logo_indossato", "topic:DC", "topic:Batman"]
"zaino con disegno Naruto" → ["logo_indossato", "topic:Anime", "topic:Naruto_anime"]
"dungeon and dragons" → ["topic:Oggetti D&D"]
"d&d" → ["topic:Oggetti D&D"]
"maglietta logo non riconosciuto" → ["logo_indossato"]
"logo non identificato sulla felpa" → ["logo_indossato"]
"cerco videogiochi" → ["topic:Videogiochi"]
"bambino con cappellino Pokemon" → ["eta_bambino", "topic:Pokemon"]
"sono gia' stato al padiglione fumetti" → ["visited:Fumetti"]
"ragazze vestite da fata; possibile manga" → ["costume_cosplay", "topic:Manga", "topic:Anime"]
"persona con maglietta retro Pac-Man" → ["logo_indossato", "topic:Cabinati Retro", "topic:Pac-Man"]
"maglietta di un anime che non conosco" → ["logo_indossato"]
"non so cosa fare" → ["nessun_indizio"]
"voglio sapere fumetti dark" → ["topic:Fumetti"]
"cerco roba dark nei fumetti" → ["topic:Fumetti"]

Restituisci SOLO un array JSON, nient'altro.

Input: {descrizione}"""

    risposta = _openai_call(system, user, temperature=0.0, max_tokens=200)
    if risposta:
        try:
            match = re.search(r'\[.*?\]', risposta, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                visive   = [o for o in parsed if is_known_observation(o) and o != "nessun_indizio"]
                topics   = [o for o in parsed if o.startswith("topic:")]
                visits   = [o for o in parsed if o.startswith("visited:")]
                buys     = [o for o in parsed if o.startswith("bought:")]
                result = visive + topics + visits + buys
                if result:
                    return result
                # Se il modello restituisce solo ["nessun_indizio"],
                # non fermarti: prova comunque i fallback simbolici.
                if parsed == ["nessun_indizio"]:
                    pass
        except Exception:
            pass

    # ── fallback keyword (senza GPT) ─────────────────────────────────
    desc_lower = descrizione.lower()
    trovate = []

    # Rileva acquisto o visita dal testo
    is_purchase = any(v in desc_lower for v in PURCHASE_VERBS)
    is_visit = any(v in desc_lower for v in VISIT_VERBS)

    if (is_purchase or is_visit) and G is not None:
        mentioned = _match_text_to_nodes(descrizione, G)
        for nid in mentioned:
            if nid in ("W.C.", "Ristoro", "robot", "Servizi"):
                continue
            if is_purchase:
                trovate.append(f"bought:{nid}")
            else:
                trovate.append(f"visited:{nid}")

    # Rileva interesse dichiarato anche senza match del modello.
    # Frasi come "voglio sapere fumetti dark" devono attivare almeno
    # la categoria generale pertinente, non cadere su nessun_indizio.
    is_interest = any(v in desc_lower for v in [
        "cerco", "vorrei", "voglio", "mi piace", "mi interessa",
        "interessa", "sono interessato", "sono interessata",
        "dove trovo", "voglio sapere", "sapere", "info su", "informazioni su"
    ])

    mapping_verbale = {

    # videogiochi
    "videogioco": "topic:Videogiochi",
    "videogame": "topic:Videogiochi",
    "gaming": "topic:Videogiochi",

    # giochi da tavolo
    "giochi da tavolo": "topic:Collezionabili Editoria",
    "boardgame": "topic:Collezionabili Editoria",
    "board game": "topic:Collezionabili Editoria",

    # fumetti
    "fumett": "topic:Fumetti",
    "comic": "topic:Fumetti",
    "comics": "topic:Fumetti",
    "manga": "topic:Manga",
    "anime": "topic:Anime",

    # cinema
    "film": "topic:Film Serie Animazione",
    "serie": "topic:Serie TV",
    "nostalgic": "topic:Film Nerd Nostalgici",
    "retro": "topic:Cabinati Retro",

    # eventi
    "concerto": "topic:Concerto",
    "concerti": "topic:Concerto",
    "musica": "topic:Concerto",
    "sigle": "topic:Concerto",
    "canzoni": "topic:Concerto",
    "giorgio vanni": "topic:Concerto",
    "cristina d'avena": "topic:Concerto",

    # collezionismo
    "collezion": "topic:Collezionabili Editoria",
    "fantasy": "topic:Libri Fantasy",
    "d&d": "topic:Oggetti D&D",
    "dnd": "topic:Oggetti D&D",
    "dungeon and dragon": "topic:Oggetti D&D",
    "dungeon and dragons": "topic:Oggetti D&D",
    "dungeons and dragons": "topic:Oggetti D&D",
    "lego": "topic:Lego",
    "funko": "topic:Funko Pop",

    # VR / eSports
    "vr": "topic:VR Arena ed eSports",
    "realtà virtuale": "topic:VR Arena ed eSports",
    "virtual reality": "topic:VR Arena ed eSports",
    "esport": "topic:VR Arena ed eSports",
    "e-sport": "topic:VR Arena ed eSports",
    "arena": "topic:VR Arena ed eSports",
    "league of legends": "topic:VR Arena ed eSports",
    "lol": "topic:VR Arena ed eSports",

    # brand
    "nintendo": "topic:Nintendo",
    "playstat": "topic:PlayStation",
    "gta": "topic:Conferenza GTA6",
    "cosplay": "topic:Sfida Cosplayer",

    # servizi
    "bagn": "topic:Servizi",
    "toilette": "topic:Servizi",
    "w.c.": "topic:W.C.",
    "wc": "topic:W.C.",
    "mangiare": "topic:Ristoro",
    "ristoro": "topic:Ristoro",
    "food": "topic:Ristoro",
    "fame": "topic:Ristoro",
    "bar": "topic:Ristoro",

    # universi
    "marvel": "topic:Marvel",
    "dc": "topic:DC",
    "batman": "topic:Batman",
    "spider": "topic:Spider-Man",
    "star wars": "topic:Star Wars",
    "stranger things": "topic:Stranger Things",
    "breaking bad": "topic:Breaking Bad",
    "zelda": "topic:Zelda",
    "mario": "topic:Mario",
    "pokemon": "topic:Pokemon",
    "minecraft": "topic:Minecraft",
    "dylan dog": "topic:Dylan Dog",
    "berserk": "topic:Berserk",
    "naruto": "topic:Naruto_anime",
    "one piece": "topic:OnePiece_anime",
    "signore degli anelli": "topic:Il Signore degli Anelli",
    "lord of the rings": "topic:Il Signore degli Anelli",
    "ghostbusters": "topic:Ghostbusters",
    "detroit": "topic:Detroit Become Human",
    "last of us": "topic:The Last of Us",
    "squid game": "topic:Squid Game",
}
    mapping_visivo = {
        "busta": "busta_acquisti",
        "costume": "costume_cosplay", "travestit": "costume_cosplay",
        "cosplay": "costume_cosplay",
        "bambino": "eta_bambino", "bambina": "eta_bambino",
        "funko": "gadget_mano", "pupazzo": "gadget_mano",
    }

    # Se non abbiamo trovato visited/bought, cerca topic e visivi
    if not trovate:
        for keyword, etichetta in {**mapping_verbale, **mapping_visivo}.items():
            if keyword in desc_lower and etichetta not in trovate:
                trovate.append(etichetta)

    # Se l'utente sta formulando una richiesta/interesse e ha nominato
    # una categoria culturale, privilegia topic:* rispetto a nessun_indizio.
    if is_interest:
        for keyword, etichetta in mapping_verbale.items():
            if keyword in desc_lower and etichetta not in trovate:
                trovate.append(etichetta)

    # ── ULTIMO FALLBACK: cerca nodi del grafo menzionati nel testo ───
    # Se non abbiamo trovato nulla, prova a matchare direttamente
    # qualsiasi nome di nodo nel testo e trattalo come topic
    if not trovate and G is not None:
        mentioned = _match_text_to_nodes(descrizione, G)
        for nid in mentioned:
            if nid in ("W.C.", "Ristoro", "robot", "Servizi"):
                continue
            tag = f"topic:{nid}"
            if tag not in trovate:
                trovate.append(tag)

    return trovate if trovate else ["nessun_indizio"]


def genera_risposta_robot(azione, target, belief, G, wm):
    tops       = top_beliefs(belief, k=3)
    belief_str = ", ".join(f"{n} ({p:.0%})" for n, p in tops if p > 0.03)
    script_str = get_script(target) if target else ""

    obs_cumulative = wm.get("_obs_cumulative", []) if wm else []

    nav_str = ""
    narrativa_str = ""
    # Le connessioni analogiche vengono passate al LLM SOLO quando
    # l'azione le richiede. Per suggerisci_stand il robot deve
    # concentrarsi sul nodo target, non distrarsi con le analogie.
    azioni_con_connessione = ("mostra_connessione", "suggerisci_adiacente")
    if target and azione in azioni_con_connessione:
        nav    = navigate(G, target, hops=1, osservazioni=obs_cumulative)
        vicini = [(n, s, r) for n, s, r in nav[:5]
                  if "vicinanza" in r and get_script(n)
                  and (wm is None or n not in wm["confermati"])]
        if vicini:
            nav_str = f"Contenuti collegati non ancora visti: {', '.join(n for n, s, r in vicini[:3])}."
            for nb, score, rel in vicini[:2]:
                for src, dst in [(target, nb), (nb, target)]:
                    if G.has_edge(src, dst):
                        narr = G[src][dst].get("narrativa")
                        if narr:
                            narrativa_str = f"Connessione curata da usare: \"{narr}\""
                            break
                if narrativa_str:
                    break

    gia_visitati_str = ""
    if wm and wm["confermati"]:
        gia_visitati_str = (f"Il visitatore ha gia' visitato: "
                            f"{', '.join(sorted(wm['confermati']))}. "
                            f"Non suggerire di nuovo queste aree.")

    turni_senza_info = wm.get("turni_senza_info", 0) if wm else 0

    istruzione = {
                "rispondi_fattuale":
            "Rispondi direttamente alla domanda del visitatore in modo fattuale. "
            "Se il contenuto non ha un padiglione autonomo, dillo chiaramente e indica la zona corretta. "
            "NON proporre connessioni analogiche, NON suggerire altri contenuti se non richiesto.",

        "suggerisci_stand":
            "Indica direttamente lo stand del NODO TARGET. Sii specifico "
            "su dove si trova. NON suggerire altri stand o contenuti — "
            "rispondi SOLO sul nodo target indicato sopra. "
            "Inizia con un riferimento all'indizio visivo osservato.",
        "suggerisci_evento":
            "Segnala l'evento con orario. Crea senso di urgenza.",
        "mostra_connessione":
            "Fai il suggerimento principale, poi aggiungi la connessione "
            "analogica fornita sotto. Riformulala con parole tue ma "
            "mantieni il contenuto culturale — non inventare connessioni diverse.",
        "suggerisci_adiacente":
            "Il visitatore ha gia' visto l'area principale. "
            "Suggerisci un'area adiacente usando la connessione culturale fornita.",
        "chiedi_mirato":
            "Fai UNA domanda al visitatore per capire se è interessato "
            "a questo argomento. Esempio: 'Ti interessa X? Abbiamo Y e Z.' "
            "Non parlare come se fossi un espositore — sei un robot guida.",
        "chiedi_aperto":
            (
                "Hai visto un indizio visivo ma non hai riconosciuto il contenuto. "
                "Dillo con naturalezza e fai una domanda aperta per capire l'interesse, "
                "per esempio chiedendo se e' piu' interessato a anime, fumetti, film/serie o videogame. "
                "Non assumere che il logo appartenga a un fandom specifico. "
                if ("logo_indossato" in obs_cumulative and not any(isinstance(o, str) and o.startswith(("topic:", "visited:", "bought:")) for o in obs_cumulative))
                else f"Fai una domanda aperta e accogliente per capire cosa cerca. "
                     f"Non ripetere domande gia' fatte (turno {wm['turno'] if wm else 1}). "
                     f"Varia la formulazione."
            ),
        "offri_mappa":
            "Il visitatore non ha ancora dato indicazioni utili. "
            "Offri una panoramica delle zone della fiera in modo invitante, "
            "come se stessi mostrando una mappa.",
    }.get(azione, "Aiuta il visitatore.")

    system = (
        "Sei un robot help-desk a Lucca Comics & Games. "
        "Sei proattivo, cordiale, conosci la fiera nei dettagli. "
        "Rispondi in italiano. Massimo 2-3 frasi."
    )
    user = (
        f"Turno: {wm['turno'] if wm else 1}\n"
        f"Azione: {azione}\n"
        f"Nodo target: {target or 'nessuno'}\n"
        f"Credenza: {belief_str}\n"
        f"Contesto: {script_str}\n"
        f"{nav_str}\n"
        f"{narrativa_str}\n"
        f"{gia_visitati_str}\n\n"
        f"Istruzione: {istruzione}\n\n"
        "Scrivi solo la risposta del robot, senza prefissi."
    )

    risposta = _openai_call(system, user)
    if risposta is None:
        nav_fallback   = navigate(G, target, hops=1) if target else []
        vicini_fallback = [n for n, s, r in nav_fallback[:3]
                           if "vicinanza" in r and get_script(n)
                           and (wm is None or n not in wm["confermati"])]
        conn_str = ", ".join(vicini_fallback) if vicini_fallback else "le zone vicine"
        fallback = {
                        "rispondi_fattuale":
                f"{target} non ha un padiglione autonomo. {script_str}",

            "suggerisci_stand":    f"Ho notato il tuo interesse per {target}! {script_str}",
            "suggerisci_evento":   f"Attenzione: {script_str}",
            "mostra_connessione":  f"Per {target}: {script_str} Guarda anche: {conn_str}.",
            "suggerisci_adiacente":f"Visto che hai gia' visitato quell'area, "
                                    f"ti consiglio: {script_str}",
            "chiedi_mirato":       f"Stai cercando qualcosa in particolare tra "
                                    f"{target} e le zone vicine?",
            "chiedi_aperto":       (
                                    "Non ho riconosciuto bene il logo: sei qui piu' per anime, fumetti, film e serie, o videogame?"
                                    if ("logo_indossato" in obs_cumulative and not any(isinstance(o, str) and o.startswith(("topic:", "visited:", "bought:")) for o in obs_cumulative))
                                    else "Benvenuto! Cosa ti interessa di piu' oggi alla fiera?"
                                ),
            "offri_mappa":         "Eccoti una panoramica: Padiglione A (Fumetti), "
                                    "B-C (Videogiochi), Piazza Anfiteatro (Film), "
                                    "Piazza Santa Maria (Collezionabili). "
                                    "Cosa ti attira di piu'?",
        }
        risposta = fallback.get(azione, "Benvenuto!")
    return risposta


_episode_memory = []


def update_weights_from_feedback(G, path_nodes, feedback, learning_rate=0.08):
    if feedback == 0 or len(path_nodes) < 2:
        return G
    n_steps = len(path_nodes) - 1
    for step, (n1, n2) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
        decay = np.exp(-0.5 * (n_steps - 1 - step))
        delta = learning_rate * feedback * decay
        for src, dst in [(n1, n2), (n2, n1)]:
            if G.has_edge(src, dst):
                old_w = G[src][dst].get("weight", 0.5)
                G[src][dst]["weight"] = round(float(np.clip(old_w + delta, 0.05, 1.0)), 4)
                break
    return G


def submit_feedback(G, observations, path_nodes, action, target,
                    confidence, belief_top5, feedback_value):
    G = update_weights_from_feedback(G, path_nodes, feedback_value)
    # Il feedback rinforza/indebolisce gli archi del grafo (già fatto sopra).
    # Poiché l'observation model è derivato dal grafo, i pesi aggiornati
    # si riflettono automaticamente nel prossimo calcolo.
    # Invalidiamo la cache perché i pesi degli archi sono cambiati.
    if feedback_value != 0:
        invalidate_obs_cache()
    _episode_memory.append({"observations": observations, "target": target,
                             "action": action, "feedback": feedback_value,
                             "path": path_nodes})
    segno = "+" if feedback_value > 0 else ""
    print(f"  [feedback] {segno}{feedback_value} registrato. "
          f"Episodi totali: {len(_episode_memory)}")
    return G


RISPOSTE_FUNZIONALI = {
    "bagno":    "I bagni sono vicino ai padiglioni principali e a Piazza Santa Maria.",
    "wc":       "I bagni sono vicino ai padiglioni principali e a Piazza Santa Maria.",
    "toilet":   "I bagni sono vicino ai padiglioni principali e a Piazza Santa Maria.",
    "mangiare": "Food truck e bar nell'area relax vicino Piazza Santa Maria.",
    "ristoro":  "Food truck e bar nell'area relax vicino Piazza Santa Maria.",
    "cibo":     "Food truck e bar nell'area relax vicino Piazza Santa Maria.",
    "fame":     "Food truck e bar nell'area relax vicino Piazza Santa Maria, aperti tutto il giorno!",
    "ho fame":  "Food truck e bar nell'area relax vicino Piazza Santa Maria, aperti tutto il giorno!",
    "mangio":   "Food truck e bar nell'area relax vicino Piazza Santa Maria.",
    "bar":      "Il bar e' nell'area relax vicino a Piazza Santa Maria.",
    "acqua":    "Il bar e' nell'area relax vicino a Piazza Santa Maria.",
    "uscit":    "L'uscita principale e' verso Piazza Napoleone.",
    "entr":     "L'entrata principale e' da Piazza Napoleone.",
    "mappa":    "Zone principali: Pad. A (Fumetti), B-C (Videogiochi), "
                "Piazza Anfiteatro (Film), Piazza Santa Maria (Collezionabili).",
}

PAROLE_CULTURALI = [
    "fumett", "manga", "anime", "videogioc", "film", "serie", "collezion",
    "event", "concerto", "cosplay", "nintendo", "playstation", "marvel",
    "dc", "lego", "fantasy", "retro", "conferenz",
]

INTERAZIONE_META = [
    "intervista", "ti sto testando", "sono un ricercatore",
    "sto studiando", "provo il sistema", "test", "speriment",
]

GIA_STATO_KEYWORDS = [
    "gia' stato", "gia stato", "ci sono stato", "l'ho visto",
    "ho gia' visto", "ho gia visto", "sono passato", "l'ho gia' visitato",
    "ci sono gia' stato", "gia' andato", "gia andato", "sono gia' andato",
    "sono gia andato", "ci sono gia' stata", "ci sono gia stata",
    "gia' visitato", "gia visitato", "l'ho gia'", "l'ho gia",
    "altri suggerimenti", "cosa altro", "cos'altro", "altro da vedere",
    "altro da suggerire", "altro?", "e poi?", "e dopo?",
]

def classifica_intento_utente(raw, wm=None):
    text = raw.lower().strip()

    ask_location_patterns = [
        "dove", "dov'e", "dov'è", "dove si trova", "in che zona",
        "in quale zona", "come ci arrivo"
    ]
    ask_existence_patterns = [
        "c'e", "c'è", "esiste", "c'e anche", "c'è anche",
        "c'e un padiglione", "c'è un padiglione",
        "c'e anche un padiglione", "c'è anche un padiglione"
    ]
    ask_confirmation_patterns = [
        "giusto", "vero", "quindi", "è lì", "e' li", "si trova li", "si trova lì"
    ]
    ask_recommendation_patterns = [
        "cosa mi consigli", "cos'altro", "altro da vedere",
        "altro da suggerire", "mi consigli", "e poi", "e dopo"
    ]

    if any(p in text for p in ask_location_patterns):
        return "ask_location"
    if any(p in text for p in ask_existence_patterns):
        return "ask_existence"
    if any(p in text for p in ask_confirmation_patterns):
        return "ask_confirmation"
    if any(p in text for p in ask_recommendation_patterns):
        return "ask_recommendation"
    return "informazione_generica"



def sessione_dialogo(G):
    print("\n" + "=" * 60)
    print("  SESSIONE DI DIALOGO CONTINUA — Robot Lucca Comics")
    print()
    print("  Scrivi liberamente quello che vedi o senti.")
    print("  +     visitatore soddisfatto, se ne va")
    print("  -     visitatore insoddisfatto")
    print("  fine  chiudi il programma")
    print("=" * 60)

    print("\n  Ora simulata (invio = ora reale, oppure 0-23):")
    ora_raw = input("  > ").strip()
    try:
        ora_sim = int(ora_raw) if ora_raw else None
    except ValueError:
        ora_sim = None

    ora_display = f"{ora_sim}:00" if ora_sim is not None \
                  else f"{datetime.datetime.now().hour}:xx (reale)"
    print(f"  [sistema] Ora: {ora_display}")

    invalidate_obs_cache()
    belief = prior_contestuale(G, ora_simulata=ora_sim)
    wm = init_working_memory()
    osservazioni_cumulative = []
    path_nodes_cumulativo = []
    ultima_azione = None
    ultimo_target = None
    ultima_confidenza = 0.0

    while True:
        wm["turno"] += 1
        print(f"\n{'='*60}")
        raw = input("  > ").strip()
        if not raw:
            continue

        raw_lower = raw.lower()

        # ── comandi speciali ─────────────────────────────────────────────
        if raw_lower == "fine":
            print("\n  Programma terminato.")
            break

        if raw == "+":
            if ultimo_target:
                G = submit_feedback(
                    G, osservazioni_cumulative, path_nodes_cumulativo,
                    ultima_azione, ultimo_target, ultima_confidenza,
                    top_beliefs(belief, 5), +1
                )
            print("  [sistema] Visitatore soddisfatto — pronto per il prossimo.\n")
            invalidate_obs_cache()
            belief = prior_contestuale(G, ora_simulata=ora_sim)
            wm = init_working_memory()
            osservazioni_cumulative = []
            path_nodes_cumulativo = []
            ultima_azione = None
            ultimo_target = None
            ultima_confidenza = 0.0
            continue

        if raw == "-":
            if ultimo_target:
                G = submit_feedback(
                    G, osservazioni_cumulative, path_nodes_cumulativo,
                    ultima_azione, ultimo_target, ultima_confidenza,
                    top_beliefs(belief, 5), -1
                )
            print("  [sistema] Feedback negativo — continua a descrivere.")
            continue

        # ── interazione meta ─────────────────────────────────────────────
        if any(k in raw_lower for k in INTERAZIONE_META):
            print(
                f"\n  ROBOT -> \"Ah, sei qui per osservarmi! Sono il robot "
                f"help-desk di Lucca Comics. Posso comunque guidarti nella "
                f"fiera se vuoi esplorare — basta dirmi cosa ti interessa.\""
            )
            continue

        # ── frasi di congedo → chiusura automatica ──────────────────────
        CONGEDO_KEYWORDS = [
            "grazie", "ok grazie", "perfetto grazie", "vado", "ci vado",
            "vado li", "vado la", "trovato", "ho capito", "capito grazie",
            "ciao", "arrivederci", "a dopo", "ok ciao", "perfetto",
        ]
        if any(k in raw_lower for k in CONGEDO_KEYWORDS):
            if ultimo_target:
                G = submit_feedback(
                    G, osservazioni_cumulative, path_nodes_cumulativo,
                    ultima_azione, ultimo_target, ultima_confidenza,
                    top_beliefs(belief, 5), +1
                )
            print(f"\n  ROBOT -> \"Prego! Buona visita a Lucca Comics!\"")
            print("  [sistema] Visitatore soddisfatto — pronto per il prossimo.\n")
            invalidate_obs_cache()
            belief = prior_contestuale(G, ora_simulata=ora_sim)
            wm = init_working_memory()
            osservazioni_cumulative = []
            path_nodes_cumulativo = []
            ultima_azione = None
            ultimo_target = None
            ultima_confidenza = 0.0
            continue

        # ── casi funzionali (solo se non misti a contenuti culturali) ───
        contiene_culturale = any(p in raw_lower for p in PAROLE_CULTURALI)
        risposta_funzionale = None
        if not contiene_culturale:
            for keyword, risposta_diretta in RISPOSTE_FUNZIONALI.items():
                if keyword in raw_lower:
                    risposta_funzionale = risposta_diretta
                    break

        if risposta_funzionale:
            print(f"\n  ROBOT -> \"{risposta_funzionale}\"")
            if "Ristoro" in belief:
                belief["Ristoro"] = 0.60
                belief["Servizi"] = 0.25
                total = sum(belief.values())
                belief = {n: v / total for n, v in belief.items()}
            ultima_azione = "funzionale"
            print("  (continua a descrivere, oppure + soddisfatto)")
            continue
        negato_specifico = trova_nodo_negato(raw, G)

        RISPOSTA_NEGATIVA_FORTE = [
            "no ",
            "no,",
            "no.",
            "non mi interessa",
            "non voglio",
            "non cerco",
            "non e' quello",
            "non è quello",
            "non e' cio'",
            "non è ciò",
            "per niente",
            "nope"
        ]

        if (
            negato_specifico
            and ("non " in raw_lower or "no" in raw_lower)
            and negato_specifico in belief
        ):
            belief[negato_specifico] *= 0.15
            total = sum(belief.values())
            belief = {n: v / total for n, v in belief.items()}
            print(f"  [belief] Nodo specifico negato: '{negato_specifico}'")
            wm["turni_senza_info"] = 0

        elif (
            ultima_azione in ("chiedi_mirato", "chiedi_aperto")
            and ultimo_target
            and any(raw_lower.startswith(k) or f" {k}" in raw_lower for k in RISPOSTA_NEGATIVA_FORTE)
        ):
            if ultimo_target in belief:
                belief[ultimo_target] *= 0.15
                total = sum(belief.values())
                belief = {n: v / total for n, v in belief.items()}
                print(f"  [belief] Risposta negativa: '{ultimo_target}' penalizzato.")
                wm["turni_senza_info"] = 0

            wm["confermati"].add(ultimo_target)
            print(f"  [wm] '{ultimo_target}' marcato come gia' visitato.")

        # ── percezione con memoria dialogica ─────────────────────────────
        referente = wm.get("referente_dialogico")

        ANAFORE_VISITA = [
    "ci sono stato", "ci sono stata",
    "ci sono gia' stato", "ci sono già stato",
    "gia' stato", "già stato",
    "l'ho visto", "l ho visto",
    "l'ho gia' visto", "l'ho già visto",
    "te l'ho detto", "te l ho detto",
    "ho detto", "ci sono stato ho detto"
]

        ANAFORE_NEGAZIONE = [
    "non quello",
    "non e' quello",
    "non è quello",
    "non mi interessa",
    "non cerco quello",
    "non quello che intendevo",
    "non ci voglio andare",
    "non voglio andarci",
    "non ci vado",
    "non mi va"
]

        if referente and any(k in raw_lower for k in ANAFORE_VISITA):
            nuove_obs = [f"visited:{referente}"]
            wm["ultimo_atto_utente"] = "visited_referente"
            print(f"  [percezione-anaphora] {nuove_obs}")

        elif referente and any(k in raw_lower for k in ANAFORE_NEGAZIONE):
            nuove_obs = [f"negated:{referente}"]
            wm["ultimo_atto_utente"] = "negated_referente"
            print(f"  [percezione-anaphora] {nuove_obs}")

        else:
            nuove_obs = percezione_da_testo(raw, G=G)
            wm["ultimo_atto_utente"] = "osservazione_normale"
            print(f"  [percezione] {nuove_obs}")
        intento_utente = classifica_intento_utente(raw, wm=wm)

        semantic_features = extract_semantic_features(raw)
        for feat in semantic_features:
            if feat not in wm["_obs_cumulative"]:
                wm["_obs_cumulative"].append(feat)

        nuove_obs, refined_topics = enrich_topics_with_instances(raw, nuove_obs, G, wm=wm)
        topic_nodes = [o.split(":", 1)[1] for o in nuove_obs if o.startswith("topic:")]
        topic_dominante = seleziona_topic_pragmatico(raw, topic_nodes, G, wm=wm)

        if topic_dominante:
            wm["topic_dominante_turno"] = topic_dominante
        else:
            wm["topic_dominante_turno"] = None

        if refined_topics:
            print(f"  [istanze pertinenti] {refined_topics}")

        # ── transazionali (legacy + visited/bought da GPT) ──────────────
        for obs in nuove_obs:
            if obs in INDIZI_TRANSAZIONALI:
                nodo_impl = TRANSAZIONALE_IMPLICA.get(obs)
                if nodo_impl:
                    mark_visited(G, nodo_impl, wm, source="transazionale")

            elif obs.startswith("bought:") or obs.startswith("visited:"):
                node_id = obs.split(":", 1)[1]
                if node_id in G.nodes():
                    source = "acquisto" if obs.startswith("bought:") else "visita"
                    mark_visited(G, node_id, wm, source=source)
        # ── belief update ────────────────────────────────────────────────
        info_ricevuta = False

        for obs in nuove_obs:
            if obs.startswith("topic:"):
                nodo_topic = obs.split("topic:", 1)[1]
                if nodo_topic in G.nodes():
                    belief = boost_topic_with_graph(belief, G, nodo_topic)
                    print(f"  [topic] '{nodo_topic}' attivato tramite il grafo.")
                    if G.nodes[nodo_topic].get("type") == "istanza":
                        for anc in get_ancestors_by_appartenenza(G, nodo_topic):
                            anc_tag = f"topic:{anc}"
                            if anc_tag not in wm["_obs_cumulative"]:
                                wm["_obs_cumulative"].append(anc_tag)
                    info_ricevuta = True

            elif obs.startswith("negated:"):
                node_id = obs.split(":", 1)[1]
                if node_id in belief:
                    belief[node_id] *= 0.02
                    # boost alternative vicine nel grafo
                    nav = navigate(G, node_id, hops=1)
                    for n, s, r in nav[:4]:
                        if n in belief and n != node_id and n not in wm["confermati"]:
                            belief[n] *= 3.0
                    total = sum(belief.values())
                    if total > 0:
                        belief = {n: v / total for n, v in belief.items()}
                    print(f"  [belief] Nodo '{node_id}' negato dall'utente.")
                    # marca come confermato per evitare che venga risuggerito
                    wm["confermati"].add(node_id)
                    print(f"  [wm] '{node_id}' escluso dai suggerimenti futuri.")
                    info_ricevuta = True

            elif obs.startswith("bought:") or obs.startswith("visited:"):
                node_id = obs.split(":", 1)[1]
                if node_id in belief:
                    belief[node_id] *= 0.10
                    nav = navigate(G, node_id, hops=1)
                    for n, s, r in nav[:3]:
                        if n in belief and "vicinanza" in r and n not in wm["confermati"]:
                            belief[n] *= 2.0
                    total = sum(belief.values())
                    if total > 0:
                        belief = {n: v / total for n, v in belief.items()}
                    info_ricevuta = True

            elif is_known_observation(obs) and obs != "nessun_indizio":
                belief = bayesian_update(belief, obs, G=G)
                info_ricevuta = True

        # ── contatore turni senza informazione utile ────────────────────
        if info_ricevuta:
            wm["turni_senza_info"] = 0
        else:
            wm["turni_senza_info"] = wm.get("turni_senza_info", 0) + 1

        osservazioni_cumulative.extend([o for o in nuove_obs if o != "nessun_indizio"])

        # alimenta _obs_cumulative per la pertinenza situata
               # alimenta _obs_cumulative per la pertinenza situata
        # include sia percetti generici sia topic:
        for o in nuove_obs:
            if o != "nessun_indizio" and o != "badge_generico" and o not in wm["_obs_cumulative"]:
                wm["_obs_cumulative"].append(o)

        # ── pianificazione ───────────────────────────────────────────────
        target_fattuale = None
        if intento_utente in ("ask_existence", "ask_location", "ask_confirmation"):
            if wm.get("topic_dominante_turno"):
                target_fattuale = wm["topic_dominante_turno"]
            elif wm.get("referente_dialogico"):
                target_fattuale = wm["referente_dialogico"]

        if target_fattuale and target_fattuale in G.nodes():
            azione = "rispondi_fattuale"
            target = target_fattuale
            confidenza = confidenza_massima(belief)[1]

        elif wm.get("topic_dominante_turno") in G.nodes():
            target_dom = wm["topic_dominante_turno"]
            script_dom = get_script(target_dom)

            if any(kw in script_dom for kw in ["ore ", "15:00", "16:30", "21:00"]):
                azione = "suggerisci_evento"
                target = target_dom
                confidenza = belief.get(target_dom, confidenza_massima(belief)[1])
            else:
                azione, target, confidenza = plan(belief, G, wm=wm)

        else:
            azione, target, confidenza = plan(belief, G, wm=wm)

        ultima_azione = azione
        ultimo_target = target
        ultima_confidenza = confidenza


        if target:
            wm["referente_dialogico"] = target
            wm["suggeriti"].add(target)
            nav = navigate(G, target, hops=2, osservazioni=wm["_obs_cumulative"])
            path_nodes_cumulativo = [target] + [n for n, s, r in nav[:4]]

        wm["ultimo_intento"] = azione

        # ── output epistemico ────────────────────────────────────────────
        stampa_stato_epistemico(
            belief, G, azione, target,
            confidenza, osservazioni_cumulative, wm
        )

        # ── risposta robot ───────────────────────────────────────────────
        risposta = genera_risposta_robot(azione, target, belief, G, wm)
        print(f"\n  ROBOT -> \"{risposta}\"")

        # ── chiusura naturale dopo azione ad alta confidenza ────────────
        if azione in (
            "suggerisci_stand", "suggerisci_evento",
            "mostra_connessione", "suggerisci_adiacente"
        ) and confidenza >= 0.50:
            print("  (scrivi 'grazie' o 'ok' per chiudere, oppure continua con nuovi indizi)")
        elif azione in ("chiedi_mirato", "chiedi_aperto", "offri_mappa"):
            print("  (rispondi liberamente, oppure + per chiudere)")
        else:
            print("  (continua a descrivere, + soddisfatto, - insoddisfatto)")

    return G

def trova_nodo_negato(raw, G):
    raw_lower = raw.lower()

    for n in G.nodes():
        if n.lower() in raw_lower:
            return n

    return None

TEST_SCENARIOS = [
    {
        "id": "T1",
        "obs": ["costume_cosplay", "topic:Anime"],
        "feedback": 0,
        "desc": "Cosplayer anime generico",
        "note": "Costume generico + topic Anime. Deve attivare Anime con Cosplayer competitivo."
    },
    {
        "id": "T2",
        "obs": ["costume_cosplay", "topic:Anime", "topic:OnePiece_anime"],
        "feedback": +1,
        "desc": "Cosplayer One Piece",
        "note": "Deve convergere su Anime/Manga con One Piece come specializzazione."
    },
    {
        "id": "T3",
        "obs": ["logo_indossato", "topic:GTA"],
        "feedback": +1,
        "desc": "Fan GTA (maglietta con logo)",
        "note": "Deve far emergere Conferenza GTA6 / PlayStation."
    },
    {
        "id": "T4",
        "obs": ["logo_indossato", "topic:Il Signore degli Anelli"],
        "feedback": +1,
        "desc": "Fan Tolkien (maglietta LOTR)",
        "note": "Deve collegare Il Signore degli Anelli, Libri Fantasy e Oggetti D&D."
    },
    {
        "id": "T5",
        "obs": ["busta_acquisti", "bought:DC"],
        "feedback": 0,
        "desc": "Ha gia' comprato DC",
        "note": "DC marcato come visitato; deve orientarsi verso DC Film o alternative."
    },
    {
        "id": "T6",
        "obs": ["logo_indossato", "topic:DC", "busta_acquisti", "bought:DC"],
        "feedback": 0,
        "desc": "Fan DC che ha gia' comprato",
        "note": "Non risuggerire DC; proporre area adiacente."
    },
    {
        "id": "T7",
        "obs": ["gadget_mano", "topic:Funko Pop"],
        "feedback": 0,
        "desc": "Ha in mano un Funko",
        "note": "Deve attivare Funko Pop / Collezionabili Editoria."
    },
    {
        "id": "T8",
        "obs": ["badge_generico"],
        "feedback": 0,
        "desc": "Visitatore senza indizi",
        "note": "Bassa confidenza: deve chiedere in modo aperto."
    },
    {
        "id": "T9",
        "obs": ["badge_generico"],
        "feedback": 0,
        "ora": 13,
        "desc": "Senza indizi a pranzo",
        "note": "Prior contestuale: Ristoro / Servizi devono emergere."
    },
    {
        "id": "T10",
        "obs": ["badge_generico"],
        "feedback": 0,
        "ora": 21,
        "desc": "Senza indizi la sera",
        "note": "Prior contestuale: Concerto / Eventi devono emergere."
    },
    {
        "id": "T11",
        "obs": ["busta_acquisti", "bought:Anime"],
        "feedback": 0,
        "desc": "Ha gia' comprato anime",
        "note": "Anime visitato; deve cercare alternative vicine."
    },
    {
        "id": "T12",
        "obs": ["eta_bambino", "topic:Nintendo"],
        "feedback": +1,
        "desc": "Bambino con indizi Nintendo",
        "note": "Deve favorire Nintendo e Pokemon / Disney Film / Anime."
    },
    {
        "id": "T13",
        "obs": ["logo_indossato", "topic:Ghostbusters"],
        "feedback": 0,
        "desc": "Visitatore con maglietta Ghostbusters",
        "note": "Frame nostalgico emerge dall'oggetto culturale: Ghostbusters → Film Nerd Nostalgici."
    },
    {
        "id": "T14",
        "obs": ["logo_indossato", "topic:Stranger Things"],
        "feedback": 0,
        "desc": "Fan Stranger Things",
        "note": "Deve attivare Stranger Things / Serie TV / Oggetti D&D."
    },
    {
        "id": "T15",
        "obs": ["costume_cosplay", "topic:Star Wars"],
        "feedback": +1,
        "desc": "Cosplayer Star Wars",
        "note": "Deve collegare Star Wars, Film Nerd Nostalgici, Lego e Sfida Cosplayer."
    },
    {
        "id": "T16",
        "obs": ["topic:VR Arena ed eSports"],
        "feedback": +1,
        "desc": "Interesse dichiarato per la VR",
        "note": "Deve attivare VR Arena ed eSports con Eventi come genitore e League of Legends come analogia."
    },
    {
        "id": "T17",
        "obs": ["logo_indossato", "topic:Berserk", "visited:Manga"],
        "feedback": 0,
        "desc": "Fan Berserk che ha gia' visitato l'area Manga",
        "note": "Manga confermato (genitore visitato) => gia_noto=True => deve proporre connessione analogica (Dylan Dog) e non ripetere lo stand Manga."
    },
    {
        "id": "T18",
        "obs": ["topic:Concerto", "topic:Naruto_anime"],
        "feedback": +1,
        "desc": "Topic doppio: concerto + Naruto",
        "note": "Due topic co-attivati. Naruto e' semanticamente dominante ma Concerto ha script con orario => deve selezionare suggerisci_evento su Concerto."
    },
    {
        "id": "T19",
        "obs": ["nessun_indizio"],
        "feedback": 0,
        "ora": 15,
        "desc": "Nessun indizio vicino alla Conferenza GTA6",
        "note": "Prior contestuale ore 15 => Conferenza GTA6 (15:00-16:00) deve emergere nella distribuzione."
    },
    {
        "id": "T20",
        "obs": ["costume_cosplay", "topic:Anime", "topic:Berserk"],
        "feedback": 0,
        "desc": "Cosplayer Berserk - test observation model dinamico",
        "note": "Tre percetti combinati. L'observation model dinamico deve produrre Berserk dominante (~42%), Manga secondo (~16%), con Dylan Dog e Inuyasha come analogie orizzontali visibili."
    }
]


def run_tests(G, verbose=True):
    print("\n" + "=" * 60)
    print("  SUITE DI TEST AUTOMATICI")
    print("=" * 60)
    for sc in TEST_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"[{sc['id']}] {sc['desc']}")
        print(f"  Nota: {sc['note']}")
        invalidate_obs_cache()
        belief = prior_contestuale(G, ora_simulata=sc.get("ora", None))
        wm     = init_working_memory()
        wm["turno"] = sc["id"]
        for obs in sc["obs"]:
            if obs in INDIZI_TRANSAZIONALI:
                nodo_impl = TRANSAZIONALE_IMPLICA.get(obs)
                if nodo_impl:
                    wm["confermati"].add(nodo_impl)
                    print(f"  [wm-auto] '{nodo_impl}' marcato come gia' visitato.")
            if obs.startswith("topic:"):
                nodo_topic = obs.split("topic:", 1)[1]
                if nodo_topic in G.nodes():
                    belief = boost_topic_with_graph(belief, G, nodo_topic)
                    print(f"  [topic] '{nodo_topic}' attivato.")
            elif obs.startswith("bought:") or obs.startswith("visited:"):
                node_id = obs.split(":", 1)[1]
                if node_id in G.nodes():
                    source = "acquisto" if obs.startswith("bought:") else "visita"
                    mark_visited(G, node_id, wm, source=source)
                    if node_id in belief:
                        belief[node_id] *= 0.10
                        nav = navigate(G, node_id, hops=1)
                        for n, s, r in nav[:3]:
                            if n in belief and "vicinanza" in r and n not in wm["confermati"]:
                                belief[n] *= 2.0
                        total = sum(belief.values())
                        if total > 0:
                            belief = {n: v / total for n, v in belief.items()}
            elif is_known_observation(obs):
                belief = bayesian_update(belief, obs, G=G)
        azione, target, confidenza = plan(belief, G, wm=wm)
        if target:
            wm["suggeriti"].add(target)
        path_nodes = []
        if target:
            nav        = navigate(G, target, hops=2)
            path_nodes = [target] + [n for n, s, r in nav[:4]]
        if verbose:
            stampa_stato_epistemico(belief, G, azione, target,
                                    confidenza, sc["obs"], wm)
        risposta = genera_risposta_robot(azione, target, belief, G, wm)
        print(f"\n  ROBOT -> \"{risposta}\"")
        if sc["feedback"] != 0:
            G = submit_feedback(G, sc["obs"], path_nodes, azione, target,
                                confidenza, top_beliefs(belief, 5), sc["feedback"])
    print("\n" + "=" * 60)
    print(f"  Test completati. Episodi registrati: {len(_episode_memory)}")
    return G


if __name__ == "__main__":
    G = initialize_system("grafolucca.json")
    print("Cosa vuoi fare?")
    print("  [1] Test automatici (scenari pre-definiti)")
    print("  [2] Demo interattiva (sessione continua)")
    scelta = input("\nScelta (1 o 2): ").strip()
    if scelta == "1":
        G = run_tests(G, verbose=True)
    else:
        G = sessione_dialogo(G)