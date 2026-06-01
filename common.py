import numpy as np
import pickle
from tensorflow.keras import layers, models # type: ignore
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes

# Hyperparamètres
N_FEATURES = 8
BATCH_SIZE = 256
LOCAL_EPOCHS = 2
NUM_CLIENTS = 2
NUM_ROUNDS = 3
NOMS_CLIENTS = ['Slice 0 : eMBB', 'Slice 1 : URLLC']

# CNN
def creer_baseline():
    model = models.Sequential()
    model.add(layers.Input(shape=(N_FEATURES, 1)))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.AveragePooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return model

# Chiffrer les poids allant du client au serveur
def chiffrer_poids(weights, pub_key):
    cipher_rsa = PKCS1_OAEP.new(pub_key)
    out = []
    for w in weights:
        raw = w.astype(np.float32).tobytes()
        k = get_random_bytes(16)
        aes = AES.new(k, AES.MODE_EAX)
        data, tag = aes.encrypt_and_digest(raw)
        enc_k = cipher_rsa.encrypt(k)
        out.append({'enc_k': enc_k, 'nonce': aes.nonce,
                    'tag': tag, 'data': data, 'shape': w.shape})
    return out

# Déchiffrer les poids du client que le serveur reçoit
def dechiffrer_poids(encrypted, priv_key):
    cipher_rsa = PKCS1_OAEP.new(priv_key)
    out = []
    for p in encrypted:
        k = cipher_rsa.decrypt(p['enc_k'])
        aes = AES.new(k, AES.MODE_EAX, nonce=p['nonce'])
        raw = aes.decrypt_and_verify(p['data'], p['tag'])
        out.append(np.frombuffer(raw, dtype=np.float32).reshape(p['shape']))
    return out

def serialiser_poids_chiffres(encrypted):
    raw = pickle.dumps(encrypted)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return [arr]

def deserialiser_poids_chiffres(arrays):
    raw = arrays[0].tobytes()
    return pickle.loads(raw)