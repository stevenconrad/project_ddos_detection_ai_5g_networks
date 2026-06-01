### Titre

Détection des attaques DDOS pour les tranches de réseau 5G à l'aide d'un modèle CNN basé sur l'apprentissage fédéré à chiffrement asymétrique.



### Contexte

&#x09;En 5G, chaque tranche fonctionne de manière indépendante avec ses propres caractéristiques en termes de débit, latence et fiabilité. Néanmoins, bien qu’améliorant la QoS globale, cette architecture expose les opérateurs à de nouvelles formes de cybermenaces telles que les attaques DDOS pouvant cibler spécifiquement certaines types de tranches risquant de gravement limiter les performances de ces dernières.

Dans un contexte 5G où des milliers d'équipements IoT sont connectés simultanément, il est primordial de mettre en place des mécanismes pour détecter et ainsi stopper ces attaques.



### Objectifs

&#x09;Notre travail s'inscrit dans une optique de renforcement de la cybersécurité des réseaux de nouvelle génération. Pour ce faire, nous allions les apports de l'intelligence artificielle aux contraintes propres des environnements 5G.

&#x09;L'objectif général est de montrer qu'il est possible de détecter efficacement les attaques DDoS dans un réseau 5G multi-slices tout en respectant la confidentialité des données de chaque slice, grâce à l'apprentissage fédéré et au chiffrement asymétrique.
De manière spécifique à terme on devrait pouvoir :

* Classifier avec précision le trafic réseau entre flux normaux et flux d’attaque
* Implémenter l'apprentissage fédéré, avec entraînement distribué sur 2 clients, et sécurisation des échanges avec mises à jour de poids.



### Outils

|Outils|Catégorie|Justification|
|-|-|-|
|Python 3.12|Langage|Langage de référence en Machine Learning|
|TensorFlow / Keras|Deep Learning|Framework de Deep Learning. On s’en sert pour définir, entraîner et évaluer notre modèle CNN|
|Pandas|Traitement des données|Manipulation et analyse de notre dataset|
|Scikit-learn|ML / Évaluation|Outils de partitionnement et de métriques d'évaluation.|
|Seaborn / Matplotlib|Visualisation|Génération de boxplots, heatmaps de corrélation et matrices de confusion pour l'analyse exploratoire des données et la visualisation des résultats.|
|Jupyter Notebook|Environnement|Environnement interactif permettant de combiner code, résultats et analyses dans un seul document|
|Dataset Kaggle DDoS 5G|Données|Dataset issu d'un test 5G réel simulant des attaques DoS/DDoS sur deux tranches réseau|
|Flower|Apprentissage fédéré|Framework de simulation d'apprentissage fédéré. Orchestre les clients et le serveur FedAvg.|
|PyCryptodome|Chiffrement|Chiffrement RSA 2048 + AES 128 des poids du modèle avant transmission.|
|Ray|Calcul distribué|Backend de parallélisation utilisé par Flower pour la simulation des clients.|

### 

### Métriques

VN : Vrai Négatif, VP : Vrai Positif, FN : Faux Négatif, FP : Faux Positif

|Métrique|Définition|Pertinence dans notre contexte|
|-|-|-|
|Accuracy|Proportion de prédictions correctes sur l'ensemble des observations.|Indicateur global de performance.|
|Precision|Proportion de flux effectivement malveillants parmi les flux classés comme attaques : VP / (VP + FP).|Donne les flux légitimes bloqués à tort|
|Recall|Proportion de flux effectivement détectés parmi tous les flux malveillants réels : VP / (VP + FN).|Donne les attaques manquées (faux négatifs).|
|F1-Score|Moyenne de la précision et du recall : 2 × (Précision × Rappel) / (Précision + Rappel).|Équilibre entre précision et rappel.|
|Specificity|Proportion de flux correctement identifiés comme bénins parmi tous les flux normaux réels : VN / (VN + FP).|Capacité du modèle à ne pas déclencher de fausses alarmes sur du trafic légitime.|



La matrice de confusion a également été utilisée pour visualiser la répartition des VP, VN, FP et FN, pour une lecture détaillée du comportement du modèle par classe.

### 

### Conception et mise en place

#### 

#### Exploration et préparation du dataset

&#x09;Le dataset utilisé est un dataset DoS/DDoS spécifique à un environnement 5G, disponible sur la plateforme Kaggle (https://www.kaggle.com/datasets/iagobs/dosddos-attacks-on-5g-networks). Il contient plusieurs millions de lignes représentant des flux réseau, chacun décrit par 11 features standardisées : Src IP, Src Port, Dst Port, Protocol, Flow Duration, Total Fwd Packets, Fwd Packet Length Std, ACK Flag Count, Fwd Seg Size Min, Slice et la variable cible label (0 = BENIGN, 1 = ATTACK).

Les étapes d'exploration ont révélé plusieurs informations importantes :

* Équilibre des classes : le dataset présente une distribution majoritairement orientée vers la classe ATTACK, reflétant le contexte de simulation d'attaques massives.
* Absence de corrélation entre les features : la heatmap de correlation n'a révélé aucune corrélation significative entre les variables, ce qui implique l'indépendance relative des features.
* Conservation des valeurs aberrantes : les boxplots ont mis en évidence des outliers sur plusieurs features (notamment Flow Duration). Ces valeurs ont délibérément été conservées, car dans notre contexte, des durées de flux extrêmes (très courtes pour les attaques DDoS, très longues pour les transferts légitimes) constituent précisément les patterns discriminants que le modèle doit apprendre.

#### 

#### Prétraitement et mise en forme

&#x09;Les features Fwd Seg, Size Min et Slice ont été exclues du jeu de features final, portant le nombre de features d'entrée à 8. Le dataset a été partitionné en 80% pour l'entraînement et 20% pour le test. Les tenseurs ont ensuite été reshapés au format (8, 1) requis par la couche Conv1D de Keras.

#### 

#### Entraînement du modèle

&#x09;Le modèle CNN a été instancié via la fonction create\_baseline() et compilé avec la fonction de perte binary\_crossentropy et l'optimiseur Adam. L'entraînement utilise 2 epochs, et un validation\_split = 0.1.
Les prédictions sur le jeu d'entraînement ont été effectuées avec un seuil de décision de 0.5 : tout score ≥ 0.5 est classé ATTACK (1), tout score < 0.5 est classé BENIGN (0).

#### 

#### Apprentissage fédéré

L'apprentissage fédéré a été implémenté selon une architecture client-serveur distribuée, composée de trois processus indépendants communicant via gRPC sur localhost:8080 à savoir un serveur central et deux clients, chacun lancé dans un terminal séparé.



##### Partition des données par tranche réseau

Les données du dataset ont été réparties entre deux clients fédérés, chacun correspondant à une tranche du réseau 5G simulé. Le client 0 prend en charge la Slice 0, représentant une tranche eMBB (Enhanced Mobile Broadband), et le client 1 prend en charge la Slice 1, représentant une tranche URLLC (Ultra-Reliable Low-Latency Communications). Chaque client dispose de données issues exclusivement de sa tranche, partitionnées en 80 % pour l'entraînement local et 20 % pour l'évaluation. La normalisation (StandardScaler) est appliquée indépendamment sur chaque client, ce qui garantit qu'il n’y a aucun grand écart entre les données.



##### Sécurisation des échanges

Deux niveaux de protection sont appliqués :

**Chiffrement du canal (TLS)** : les échanges entre clients et serveur transitent dans un canal TLS activé nativement par Flower via gRPC sécurisé. Les certificats ont été générés avec une autorité de certification locale,
garantissant le chiffrement du transport dans les deux sens.

**Chiffrement des poids (RSA + AES)** : avant toute transmission des mises à jour de poids au serveur, chaque client applique un chiffrement hybride en deux couches. Pour chaque tenseur, une clé AES-128 aléatoire est générée et chiffre les données via le mode EAX, garantissant confidentialité et intégrité. La clé AES est ensuite chiffrée avec la clé publique RSA-2048 du serveur via PKCS1-OAEP et seul le serveur, détenteur de la clé privée, peut déchiffrer les poids reçus. Les clés publiques sont échangées via des fichiers .pem partagés au démarrage de la simulation.



##### Architecture client Flower (ClientSlice5G)

Chaque client est instancié à partir de la classe ClientSlice5G, qui hérite de fl.client.NumPyClient. Son cycle de vie se déroule comme suit à chaque round fédéré :

* Réception du modèle global : le client charge les poids agrégés issus du round précédent dans son instance locale du modèle CNN via set\_weights().
* Entraînement local : le modèle est entraîné sur les données locales du client. Les métriques (loss, accuracy, val\_loss, val\_accuracy) sont enregistrées.
* Chiffrement et transmission : les poids mis à jour sont chiffrés puis renvoyés au serveur accompagnés des métriques d'entraînement.
* Évaluation locale : à la demande du serveur, le client évalue le modèle global reçu sur son jeu de test local et retourne la perte et l'accuracy.
* 

### Serveur Flower et algorithme FedAvg

Le serveur utilise une stratégie FedAvgLog. À chaque round, le serveur :

* Distribue le modèle global courant aux deux clients.
* Collecte les mises à jour de poids après déchiffrement.
* Agrège les poids en les pondérant par le nombre d'exemples de chaque client. w\_global = Σ(nk × wk) / Σnk
* Enregistre les métriques agrégées (train\_loss, train\_acc, val\_loss, val\_acc) pour chaque round, afin de suivre la convergence du modèle global.



##### Paramètres de simulation

NUM\_CLIENTS = 2, NUM\_ROUNDS = 3, LOCAL\_EPOCHS = 1, BATCH\_SIZE = 256

#### Architecture

```
project\_ddos\_detection\_ai\_5g/
│
├── certificats/
│   ├── ca.pem                      Autorité de certification
│   ├── cert.pem                    Certificat serveur signé par CA
│   └── key.pem                     Clé privée serveur (TLS)
│
├── cles/
│   ├── cle\_pub\_serveur.pem         Clé publique RSA serveur
│   ├── cle\_pub\_client\_0.pem        Clé publique RSA client 0
│   └── cle\_pub\_client\_1.pem        Clé publique RSA client 1
│
├── common.py                       Hyperparamètres, CNN, fonctions chiffrement
├── server.py                       Serveur Flower + FedAvgLog + agrégation
├── client0.py                      Client Slice 0
├── client1.py                      Client Slice 1
├── evaluate.ipynb                  Cross validation + graphiques post-simulation
├── generate\_certs.py               Génération certificats TLS (une seule fois)
├── EDA.ipynb                       Exploration, préparation des données et confusion baseline
│
├── 5g\_ddos.csv                     Dataset trafic réseau 5G
│
├── poids\_federes\_finaux.npy        Poids sauvegardés après simulation
└── hist\_global.json                Historique métriques par round
```

#### Ordre de lancement

* 1 - terminal x : python generate\_certs.py
* 2 - Terminal 1 :
python server.py
* 3 - Terminal 2 :
python client0.py
* 4 -Terminal 3 :
python client1.py
* 5 -après la fin de la simulation, on exécute le notebook evaluate.ipynb cellule après cellule

