import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from Crypto.PublicKey import RSA
from common import *
from pathlib import Path
import numpy as np
import json

# Clés du serveur 
cle_serveur = RSA.generate(2048)
CLE_PUBLIQUE_SERVEUR_BYTES = cle_serveur.publickey().export_key()
CLE_PRIVEE_SERVEUR = cle_serveur

# Sauvegarder la clé publique serveur dans un fichier
with open('cles/cle_pub_serveur.pem', 'wb') as fichier:
    fichier.write(CLE_PUBLIQUE_SERVEUR_BYTES)

# Stockage des clés publiques clients et poids chiffrés
POIDS_FEDERES_FINAUX = None
HIST_GLOBAL = {
    'round': [],
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
}

class FedAvgLog(FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        # 1. Déchiffrer les poids reçus de chaque client
        resultats_dechiffres = []
        for client_proxy, fit_res in results:
            poids_chiffres = deserialiser_poids_chiffres(fl.common.parameters_to_ndarrays(fit_res.parameters) )
            poids_clairs = dechiffrer_poids(poids_chiffres, CLE_PRIVEE_SERVEUR)
            fit_res_clair = fl.common.FitRes(
                status = fit_res.status,
                parameters = fl.common.ndarrays_to_parameters(poids_clairs),
                num_examples = fit_res.num_examples,
                metrics = fit_res.metrics
            )
            resultats_dechiffres.append((client_proxy, fit_res_clair))

        # 2. FedAvg
        agg = super().aggregate_fit(server_round, resultats_dechiffres, failures)

        if agg:
            parameters_agg, _ = agg
            global POIDS_FEDERES_FINAUX
            poids_agreges = fl.common.parameters_to_ndarrays(parameters_agg)
            POIDS_FEDERES_FINAUX = poids_agreges

            # Métriques
            total = sum(fit_res.num_examples for _, fit_res in resultats_dechiffres)
            if total > 0:
                train_loss = sum(fit_res.num_examples * fit_res.metrics.get('train_loss', 0)
                                for _, fit_res in resultats_dechiffres) / total
                train_acc = sum(fit_res.num_examples * fit_res.metrics.get('train_acc', 0)
                                for _, fit_res in resultats_dechiffres) / total
                val_loss = sum(fit_res.num_examples * fit_res.metrics.get('val_loss', 0)
                                for _, fit_res in resultats_dechiffres) / total
                val_acc = sum(fit_res.num_examples * fit_res.metrics.get('val_acc', 0)
                                for _, fit_res in resultats_dechiffres) / total

                HIST_GLOBAL['round'].append(server_round)
                HIST_GLOBAL['train_loss'].append(train_loss)
                HIST_GLOBAL['train_acc'].append(train_acc)
                HIST_GLOBAL['val_loss'].append(val_loss)
                HIST_GLOBAL['val_acc'].append(val_acc)

                print(f'  [SERVEUR] Round {server_round}/{NUM_ROUNDS} : '
                    f'train_loss = {train_loss:.4f} train_acc = {train_acc:.4f} | '
                    f'val_loss = {val_loss:.4f} val_acc = {val_acc:.4f}')
            return agg

    # Aggrège l'évaluation des métriques
    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        if agg:
            loss_agregee, metrics = agg
            print(f'[SERVEUR EVAL] Round {server_round} : '
                  f'loss = {loss_agregee:.4f} accuracy = {metrics["accuracy"]:.4f}')
        return agg

# Aggrégation manuelle des accuracies car Flower ne le fait pas nativement
def moyenne_poids(metrics):
    accs = [n * m['accuracy'] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {'accuracy': sum(accs) / total}

# Paramètres initiaux que les clients reçoivent pour l'entrainement
parametres_initiaux = ndarrays_to_parameters(creer_baseline().get_weights())

strategie = FedAvgLog(
    initial_parameters = parametres_initiaux,
    fraction_fit = 1.0,
    fraction_evaluate = 1.0,
    min_fit_clients = NUM_CLIENTS,
    min_evaluate_clients = NUM_CLIENTS,
    min_available_clients = NUM_CLIENTS,
    evaluate_metrics_aggregation_fn = moyenne_poids,
)

if __name__ == '__main__':
    print("Démarrage du serveur sur localhost:8080...")
    fl.server.start_server(
        server_address = "localhost:8080",
        config = ServerConfig(num_rounds = NUM_ROUNDS),
        strategy = strategie,
        certificates=(
            Path("certificats/ca.pem").read_bytes(),     # autorité de certification
            Path("certificats/cert.pem").read_bytes(),   # certificat serveur
            Path("certificats/key.pem").read_bytes(),    # clé privée serveur
        )
    )

# Sauvegarder les poids fédérés finaux
if POIDS_FEDERES_FINAUX is not None:
    np.save('poids_federes_finaux.npy', 
            np.array(POIDS_FEDERES_FINAUX, dtype=object), 
            allow_pickle=True)
    print("Poids sauvegardés dans poids_federes_finaux.npy")

    with open('hist_global.json', 'w') as f:
        json.dump(HIST_GLOBAL, f)
    print("Historique sauvegardé dans hist_global.json")
    