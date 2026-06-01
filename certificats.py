# generate_certs.py
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import datetime

backend = default_backend()

# ── 1. Clé et certificat CA ──
cle_ca = rsa.generate_private_key(65537, 2048, backend)
nom_ca = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"MyCA")])
cert_ca = (
    x509.CertificateBuilder()
    .subject_name(nom_ca)
    .issuer_name(nom_ca)
    .public_key(cle_ca.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    .sign(cle_ca, hashes.SHA256(), backend)
)

# ── 2. Clé et certificat SERVEUR signé par la CA ──
cle_serveur = rsa.generate_private_key(65537, 2048, backend)
nom_serveur = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"localhost")])
cert_serveur = (
    x509.CertificateBuilder()
    .subject_name(nom_serveur)
    .issuer_name(nom_ca)
    .public_key(cle_serveur.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
        critical=False
    )
    .sign(cle_ca, hashes.SHA256(), backend)
)

# ── 3. Sauvegarder ──
with open("certificats/ca.pem", "wb") as f:
    f.write(cert_ca.public_bytes(serialization.Encoding.PEM))

with open("certificats/cert.pem", "wb") as f:
    f.write(cert_serveur.public_bytes(serialization.Encoding.PEM))

with open("certificats/key.pem", "wb") as f:
    f.write(cle_serveur.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()
    ))

print("Certificats générés :")
print("  ca.pem : autorité de certification")
print("  cert.pem : certificat serveur signé par CA")
print("  key.pem : clé privée serveur")