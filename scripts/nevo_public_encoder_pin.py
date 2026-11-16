"""Load public sentence encoder by immutable revision, for inference ONLY."""
from huggingface_hub import HfApi,snapshot_download
import json, pathlib
model = "sentence-transformers/all-MiniLM-L12-v2"
sha = "a50ef00143b4d5391434df20ae11632588ac25be"
info = HfApi().model_info(repo_id=model,revision=sha)
assert info.sha == sha
snapshot = pathlib.Path(snapshot_download(repo_id=model,revision="main"))
assert snapshot.name == sha, ("Mutable main moved, refusing substitution", snapshot.name)
card=info.cardData
license = card.get("license") if isinstance(card,dict) else getattr(card,"license",None)
assert license == "apache-2.0"
print("NEVO_PUBLIC_ENCODER_PROVENANCE",json.dumps({
    "public_model":model,"revision":sha,"license":license,
    "snapshot_sha_verified":True,"training_updates":0
},sort_keys=True),flush=True)
