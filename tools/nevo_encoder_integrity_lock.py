#!/usr/bin/env python3
"""D170-R2R public Nevo encoder assets: immutable, fail-closed native-loader verification.

No edits to Nevo-CRL source. Stage the exact revision in the layout used by
sentence-transformers 2.2.2, verify upstream file checksums and every local
byte, negative-test corruption/revision/missing files, and execute the
unchanged original learner offline. No optimization or neural training.
"""
import hashlib
import json
import os
import pathlib
import runpy
import shutil
import sys
import tempfile
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent
LOCK = REPO / ".runtime_nevo_encoder_lock.json"
MODEL = "sentence-transformers/all-MiniLM-L12-v2"
SHA = "a50ef00143b4d5391434df20ae11632588ac25be"


class IntegrityFailure(RuntimeError):
    pass


def insist(condition, msg):
    if not condition:
        raise IntegrityFailure(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def loader_root():
    import torch
    # Directly matches the original sentence-transformers v2.2.2 constructor.
    return (pathlib.Path(torch.hub._get_torch_home()) /
            "sentence_transformers" / "sentence-transformers_all-MiniLM-L12-v2").resolve()


def actual_assets(base):
    insist(base.is_dir(), "Original model folder missing")
    found = {}
    for p in base.rglob("*"):
        insist(not p.is_symlink(), "Unverified model symlink: " + str(p))
        if p.is_file():
            name = p.relative_to(base).as_posix()
            found[name] = {"sha256": sha256(p), "size": p.stat().st_size}
    return found


def verify(lock):
    insist(lock["repo"] == MODEL, "Wrong public model")
    insist(lock["revision"] == SHA, "Wrong model revision")
    insist(lock["legacy_loader_root"] == str(loader_root()), "Unexpected model path")
    expected = lock["assets"]
    seen = actual_assets(loader_root())
    insist(set(expected) == set(seen), "Missing/extra model files: " + str({
        "missing": sorted(set(expected).difference(seen)),
        "extra": sorted(set(seen).difference(expected))
    }))
    for name in expected:
        insist(seen[name] == expected[name], "Hash/size mismatch: " + name)
    insist("modules.json" in seen, "Missing modules.json")
    return seen


def must_reject(case, action):
    try:
        action()
    except IntegrityFailure as exc:
        print("ENCODER_NEGATIVE_FAIL_CLOSED_PASS", json.dumps({
            "case": case, "reason": str(exc)[:240]
        }, sort_keys=True), flush=True)
        return
    raise IntegrityFailure("FAIL-OPEN negative test: " + case)


def prepare():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo_id=MODEL, revision=SHA, files_metadata=True)
    insist(info.sha == SHA, "Hub returned unexpected revision")
    card = info.cardData
    lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    insist(lic == "apache-2.0", "Model license metadata differs")
    snapshot = pathlib.Path(snapshot_download(repo_id=MODEL, revision=SHA))
    insist(snapshot.name == SHA, "Snapshot did not resolve immutable revision")
    remote = {item.rfilename: item for item in info.siblings}
    local = {p.relative_to(snapshot).as_posix(): p
             for p in snapshot.rglob("*") if p.is_file()}
    insist(set(local) == set(remote), "Incomplete pinned model snapshot")
    destination = loader_root()
    insist(not destination.exists(), "Refusing pre-existing unverified legacy cache")
    destination.mkdir(parents=True, exist_ok=False)

    records = {}
    lfs_checked = 0
    git_checked = 0
    for name, src in sorted(local.items()):
        rel = pathlib.PurePosixPath(name)
        insist(not rel.is_absolute() and ".." not in rel.parts, "Unsafe asset path")
        dst = destination.joinpath(*rel.parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(dst))
        digest = sha256(dst)
        insist(digest == sha256(src), "Copy integrity mismatch: " + name)
        size = dst.stat().st_size
        remote_file = remote[name]
        lfs = getattr(remote_file, "lfs", None)
        if lfs:
            upstream_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            insist(upstream_sha == digest, "Git LFS checksum mismatch: " + name)
            lfs_checked += 1
        else:
            upstream_git = getattr(remote_file, "blob_id", None)
            if upstream_git:
                gh = hashlib.sha1()
                gh.update(("blob %d\0" % size).encode("ascii"))
                with open(str(dst), "rb") as f:
                    for chunk in iter(lambda: f.read(1048576), b""):
                        gh.update(chunk)
                insist(gh.hexdigest() == upstream_git, "Git blob checksum mismatch: " + name)
                git_checked += 1
        records[name] = {"sha256": digest, "size": size}

    insist(lfs_checked >= 1, "Upstream LFS digest unavailable for original weights")
    weight_names = [x for x in records
                    if x.endswith(("pytorch_model.bin", "model.safetensors"))]
    insist(weight_names, "Original encoder pretrained weights not present")
    lock = {"schema": "D170-R2R/encoder-lock-v1",
            "repo": MODEL, "revision": SHA,
            "legacy_loader_root": str(destination), "assets": records,
            "upstream_lfs_verified_files": lfs_checked,
            "upstream_git_blobs_verified_files": git_checked}
    verify(lock)

    weight = destination.joinpath(*pathlib.PurePosixPath(sorted(weight_names)[0]).parts)
    with open(str(weight), "r+b") as f:
        old = f.read(1)
        insist(bool(old), "Empty pretrained checkpoint")
        try:
            f.seek(0)
            f.write(bytes([old[0] ^ 1]))
            f.flush()
            must_reject("changed_original_pretrained_weight", lambda: verify(lock))
        finally:
            f.seek(0)
            f.write(old)
            f.flush()
    verify(lock)

    mods = destination / "modules.json"
    with tempfile.TemporaryDirectory() as tmp:
        saved = pathlib.Path(tmp) / "modules.json"
        mods.rename(saved)
        try:
            must_reject("missing_original_modules_file", lambda: verify(lock))
        finally:
            saved.rename(mods)
    verify(lock)

    wrong = dict(lock)
    wrong["revision"] = "0" * 40
    must_reject("incorrect_model_revision", lambda: verify(wrong))
    verify(lock)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    summary_hash = hashlib.sha256(json.dumps(
        records, separators=(",", ":"), sort_keys=True).encode("utf8")).hexdigest()
    print("NEVO_PINNED_ENCODER_ASSETS_PREPARED_PASS", json.dumps({
        "model": MODEL, "model_sha": SHA, "asset_count": len(records),
        "original_weights": sorted(weight_names),
        "verified_upstream_lfs": lfs_checked,
        "verified_upstream_git_blobs": git_checked,
        "all_assets_sha256": summary_hash,
        "corrupted_file_rejected": True,
        "missing_file_rejected": True,
        "wrong_revision_rejected": True,
        "training_updates": 0
    }, sort_keys=True), flush=True)


def execute():
    insist(LOCK.is_file(), "Missing pinned encoder manifest")
    lock = json.loads(LOCK.read_text())
    verify(lock)
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "D4RL_SUPPRESS_IMPORT_ERROR": "1"
    })
    import torch
    observed = set()
    base = loader_root()

    def audit(event, args):
        if event == "open" and args:
            try:
                resolved = pathlib.Path(os.fsdecode(args[0])).resolve()
                observed.add(resolved.relative_to(base).as_posix())
            except (ValueError, TypeError, OSError):
                pass
    sys.addaudithook(audit)
    # This is the previously qualified native Nevo constructor diagnostic:
    # full original 1024x4 architecture, five actors, no training.
    diagnostic = REPO / "scripts" / "nevo_full_native_cpu.py"
    insist(diagnostic.is_file(), "Native learner source diagnostic missing")
    scope = runpy.run_path(str(diagnostic), run_name="__main__")
    learner = scope.get("learner")
    insist(learner is not None, "Original learner not constructed")
    insist("modules.json" in observed, "Original loader did not read pinned modules.json")
    verify(lock)

    # Check actual initialized model tensors against the exact loaded weights.
    # Unlike a preflight model-info SHA, this tests the model in learner memory.
    transformer = learner.task_encoder[0]
    native = transformer.auto_model
    cfg_path = pathlib.Path(str(native.config._name_or_path)).resolve()
    insist(cfg_path == base or base in cfg_path.parents,
           "Actual transformer loaded from outside locked folder: " + str(cfg_path))
    candidates = sorted(
        (x for x in lock["assets"] if x.endswith("pytorch_model.bin")),
        key=lambda x: (len(x), x))
    insist(bool(candidates), "Missing original PyTorch state dict")
    selected = None
    pretrained = None
    for name in candidates:
        path = base.joinpath(*pathlib.PurePosixPath(name).parts)
        raw = torch.load(str(path), map_location="cpu")
        if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
            raw = raw["state_dict"]
        if not isinstance(raw, dict):
            continue
        keys = set(raw).intersection(native.state_dict())
        if len(keys) >= 30:
            selected = name
            pretrained = raw
            break
    insist(selected is not None, "No compatible original pretrained checkpoint")
    actual = native.state_dict()
    common = sorted(set(pretrained).intersection(actual))
    insist(len(common) >= 30, "Insufficient checked native tensors")
    different = [name for name in common
                 if not torch.is_tensor(pretrained[name])
                 or pretrained[name].shape != actual[name].shape
                 or not torch.equal(pretrained[name].cpu(), actual[name].cpu())]
    insist(not different, "Loaded original encoder tensors differ: " + str(different[:5]))
    insist(learner.step == 0 and all(a.step == 0 for a in learner.actor_pop),
           "Unexpected learner training step")
    verify(lock)
    summary_hash = hashlib.sha256(json.dumps(
        lock["assets"], separators=(",", ":"), sort_keys=True).encode("utf8")).hexdigest()
    print("NEVO_ACTUALLY_LOADED_ENCODER_ASSETS_PASS", json.dumps({
        "original_model_revision": SHA, "sha256_asset_manifest": summary_hash,
        "all_asset_count": len(lock["assets"]),
        "loaded_original_pretrained_checkpoint": selected,
        "matching_loaded_native_tensors": len(common),
        "differing_loaded_native_tensors": 0,
        "original_loader_read_modules_json": True,
        "native_config_path_inside_verified_root": True,
        "offline_alias_resolution": True,
        "post_execution_asset_hashes_unchanged": True,
        "negative_checks_passed": 3, "neural_optimizer_updates": 0
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "execute"):
        raise SystemExit("Usage: nevo_encoder_integrity_lock.py prepare|execute")
    try:
        {"prepare": prepare, "execute": execute}[sys.argv[1]]()
    except Exception as exc:
        print("NEVO_ENCODER_FAIL_CLOSED", json.dumps({
            "phase": sys.argv[1], "exception": type(exc).__name__,
            "error": str(exc)[:1200], "optimizer_updates": 0,
            "stack_tail": traceback.format_exc(limit=5)[-1300:]
        }, sort_keys=True), flush=True)
        raise
