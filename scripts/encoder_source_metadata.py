"""Resolve original public sentence-model SHA from its authoritative registry.
Metadata only; no training, no fine-tuning, no private requests.
"""
import json
from huggingface_hub import HfApi
m=HfApi().model_info(repo_id="sentence-transformers/all-MiniLM-L12-v2",revision="main")
license_tag=m.cardData.get("license") if isinstance(m.cardData,dict) else getattr(m.cardData,"license",None)
print("PUBLIC_ENCODER_REPOSITORY",json.dumps({
"model_id":"sentence-transformers/all-MiniLM-L12-v2",
"resolved_commit":m.sha,"license_metadata":license_tag,
"training_updates":0},sort_keys=True),flush=True)
assert len(m.sha)==40
