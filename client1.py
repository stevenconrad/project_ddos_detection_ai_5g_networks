import flwr as fl
from Crypto.PublicKey import RSA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from common import *
from pathlib import Path

# Clé publique du serveur (à copier depuis server.py)
CLE_PUB_SERVEUR_BYTES = open('cles/cle_pub_serveur.pem', 'rb').read()

CLIENT_ID = 1

class ClientSlice5G(fl.client.NumPyClient):

    def __init__(self):
        self.cid = CLIENT_ID
        self.model = creer_baseline()

        # Paire de clés client
        cle_client = RSA.generate(2048)
        self.cle_priv_client_bytes = cle_client.export_key()
        self.cle_pub_client_bytes = cle_client.publickey().export_key()

        # Sauvegarder la clé publique pour que le serveur la récupère
        with open(f'cles/cle_pub_client_{CLIENT_ID}.pem', 'wb') as f:
            f.write(self.cle_pub_client_bytes)

        # Lecture des données et suppression des lignes contenant des données manquantes
        data = pd.read_csv('5g_ddos.csv').dropna() 

        # Sélection des données de la slice
        slice_data = data[data['Slice'] == CLIENT_ID]

        # Slices non pertinentes supprimées 
        cols_drop = ['label', 'Fwd Seg Size Min', 'Slice', 'index']   
        FEATURE_COLS = [c for c in data.drop(cols_drop, axis=1).columns] 

        # Sélection des features et target
        Xc = slice_data[FEATURE_COLS].values.astype(np.float32) 
        Yc = slice_data['label'].values.astype(np.float32)

        # Séparation des données (entrainement et validation)
        X_tr, X_te, Y_tr, Y_te = train_test_split(Xc, Yc, test_size=0.2, random_state=42)
        
        # Normalisation des données
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Reshape
        self.X_train = X_tr.reshape(-1, N_FEATURES, 1)
        self.X_test = X_te.reshape(-1, N_FEATURES, 1)
        self.Y_train = Y_tr
        self.Y_test = Y_te

    # Récupérer les poids actuels du modèle
    def get_parameters(self, config):
        return self.model.get_weights() 

    # Mettre à jour les poids
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
        
    # Entrainement du modèle sur les données de la slice
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        h = self.model.fit(
            self.X_train, self.Y_train,
            epochs = LOCAL_EPOCHS,
            batch_size = BATCH_SIZE,
            validation_split = 0.1,
            verbose = 1,
            # Le dataset étant déséquilibré (Contient environ 8 fois plus de traffic DDOS que de traffic normal), on met sur pieds un mécanisme de reward
            class_weight = {0: 8, 1: 1} 
        )
        acc = h.history['accuracy'][-1]
        loss = h.history['loss'][-1]
        val_acc = h.history['val_accuracy'][-1]
        val_loss = h.history['val_loss'][-1]

        cle_pub_server = RSA.import_key(CLE_PUB_SERVEUR_BYTES)
        poids_chiffres = chiffrer_poids(self.model.get_weights(), cle_pub_server)
        poids_serialises = serialiser_poids_chiffres(poids_chiffres)

        print(f'  [Client {self.cid}] loss={loss:.4f} acc={acc:.4f}')

        return poids_serialises, len(self.X_train), {
            'train_loss': float(loss),
            'train_acc' : float(acc),
            'val_loss' : float(val_loss),
            'val_acc' : float(val_acc),
            'client_id' : float(self.cid),
        }

    # Evaluer les poids globaux sur les données de la slice 
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        return loss, len(self.X_test), {'accuracy': float(acc)}

if __name__ == '__main__':
    print(f"Démarrage client {CLIENT_ID} connexion à localhost:8080...")
    fl.client.start_client(
        server_address="localhost:8080",
        client=ClientSlice5G().to_client(),
        root_certificates=Path("certificats/ca.pem").read_bytes()
    )