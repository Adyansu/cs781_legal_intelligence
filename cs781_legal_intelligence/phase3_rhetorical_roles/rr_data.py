"""CS781 Phase 3 (Rhetorical Roles) - data discovery, format detection and validation.

The exact file layout of Phase_3_RR/ is detected at runtime, never assumed. Supported shapes
(JSON, JSONL, CSV/TSV), per split (train / dev / test) and domain (CL / IT):
  * {doc_id: {"sentences": [...], "labels": [...]}}             (any common key names)
  * [{"id": ..., "sentences": [...], "labels": [...]}, ...]
  * {doc_id: [sentence, ...]}  +  {doc_id: [label, ...]}        (text and truth in separate files)
  * {doc_id: [{"text"/"sentence": ..., "label": ...}, ...]}
  * legal-eval style [{"id", "annotations": [{"result": [{"value": {"text", "labels": [...]}}]}]}]
  * CSV/TSV rows with a document-id column, a sentence column and (optionally) a label column
Split and domain come from the file path (e.g. .../CL/train.json, it_dev.json) or a record field.
Anything unrecognised stops with an explicit message instead of guessing.
"""
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent",
          "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt",
          "RatioOfTheDecision", "RulingByPresentCourt", "None"]
LABEL_SET = set(LABELS)
DOMAINS = ["CL", "IT"]
EXPECTED_SENTENCES = {("train", "CL"): 10593, ("train", "IT"): 6252, ("dev", "CL"): 1396,
                      ("dev", "IT"): 746, ("test", "CL"): 1339, ("test", "IT"): 858}

ID_KEYS = ("id", "doc_id", "docid", "document_id", "documentid", "doc", "name", "file", "filename",
           "file_name", "case_id", "caseid", "doc_name")
SENT_KEYS = ("sentences", "sents", "sentence", "texts", "text", "segments", "lines")
LABEL_KEYS = ("labels", "label", "roles", "role", "rhetorical_roles", "rhetorical_role", "rr", "tags",
              "gold", "y")
DOMAIN_KEYS = ("domain", "category", "area", "subset")
CONTAINER_KEYS = ("data", "documents", "docs", "items", "records", "examples")


class Phase3FormatError(RuntimeError):
    pass


def _tokens(path):
    return [t for t in re.split(r"[^a-z0-9]+", path.lower()) if t]


def _split_hits(text):
    hits = set()
    for t in _tokens(text):
        if t.startswith("train"):
            hits.add("train")
        elif t in ("dev", "val", "valid", "validation", "development") or t.startswith("dev"):
            hits.add("dev")
        elif t.startswith("test"):
            hits.add("test")
    return hits


def split_from_path(path):
    """The file name decides first (Train/CL_dev.json is dev), then the folders, innermost first."""
    parts = path.replace("\\", "/").split("/")
    for part in [os.path.splitext(parts[-1])[0]] + parts[-2::-1]:
        hits = _split_hits(part)
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            return None
    return None


def domain_from_text(s):
    toks = _tokens(str(s))
    hits = set()
    for t in toks:
        if t in ("cl", "corporate", "corporatelaw", "company") or t.startswith("cl") and t[2:].isdigit():
            hits.add("CL")
        elif t in ("it", "infotech", "informationtechnology") or t.startswith("it") and t[2:].isdigit():
            hits.add("IT")
    return hits.pop() if len(hits) == 1 else None


def _first(d, keys):
    low = {k.lower(): k for k in d}
    for k in keys:
        if k in low:
            return d[low[k]]
    return None


def _is_label_list(v):
    return isinstance(v, list) and v and all(isinstance(x, str) for x in v) and all(x in LABEL_SET for x in v)


def _norm_label(x):
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    if isinstance(x, dict):
        x = _first(x, LABEL_KEYS)
        return _norm_label(x)
    return x


def _parse_sentence_dicts(items):
    sents, labs = [], []
    for it in items:
        s = _first(it, ("text", "sentence", "sent", "content", "line"))
        if s is None and "value" in it and isinstance(it["value"], dict):         # legal-eval result item
            s = it["value"].get("text")
            labs.append(_norm_label(it["value"].get("labels")))
        else:
            labs.append(_norm_label(_first(it, LABEL_KEYS)))
        sents.append(s)
    labs = None if all(l is None for l in labs) else labs
    return sents, labs


def parse_record(r, default_id=None):
    """-> (doc_id, sentences or None, labels or None, domain or None)"""
    did = _first(r, ID_KEYS)
    did = default_id if did is None else did
    dom = _first(r, DOMAIN_KEYS)
    dom = domain_from_text(dom) if dom is not None else None
    sents = labs = None
    ann = r.get("annotations")
    if isinstance(ann, list) and ann and isinstance(ann[0], dict) and isinstance(ann[0].get("result"), list):
        sents, labs = _parse_sentence_dicts(ann[0]["result"])
    else:
        sv = _first(r, SENT_KEYS)
        if isinstance(sv, dict):                                      # e.g. {"text": {...}} nested
            return parse_record({**sv, **{k: v for k, v in r.items() if k not in SENT_KEYS}}, did)
        if isinstance(sv, list) and sv and all(isinstance(x, dict) for x in sv):
            sents, labs = _parse_sentence_dicts(sv)
        elif isinstance(sv, list) and all(isinstance(x, str) for x in sv):
            sents = sv
        lv = _first(r, LABEL_KEYS)
        if isinstance(lv, list):
            labs = [_norm_label(x) for x in lv]
    if sents is not None and _is_label_list(sents) and labs is None:      # a truth file keyed under "text"
        sents, labs = None, sents
    return (None if did is None else str(did)), sents, labs, dom


def parse_obj(obj):
    out = []
    if isinstance(obj, dict):
        cont = _first(obj, CONTAINER_KEYS)
        if isinstance(cont, list) and len(obj) <= 3:
            return parse_obj(cont)
        for k, v in obj.items():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                out.append((str(k), None, v, None) if _is_label_list(v) else (str(k), v, None, None))
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                s, l = _parse_sentence_dicts(v)
                out.append((str(k), s, l, None))
            elif isinstance(v, dict):
                out.append(parse_record(v, default_id=k))
            elif isinstance(v, list) and not v:
                out.append((str(k), [], None, None))
            else:
                raise Phase3FormatError(f"unrecognised value type for key {k!r}: {type(v).__name__}")
    elif isinstance(obj, list):
        for i, r in enumerate(obj):
            if not isinstance(r, dict):
                raise Phase3FormatError(f"list item {i} is {type(r).__name__}, expected an object")
            out.append(parse_record(r))
    else:
        raise Phase3FormatError(f"top-level JSON is {type(obj).__name__}")
    return out


def parse_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        rows = list(csv.DictReader(f, dialect=dialect))
    if not rows:
        return []
    cols = {c.lower(): c for c in rows[0]}
    idc = next((cols[k] for k in ID_KEYS if k in cols), None)
    sc = next((cols[k] for k in ("sentence", "text", "sent", "sentences") if k in cols), None)
    lc = next((cols[k] for k in LABEL_KEYS if k in cols), None)
    if idc is None or sc is None:
        raise Phase3FormatError(f"{path}: CSV needs a document-id and a sentence column, got {list(rows[0])}")
    docs = defaultdict(lambda: ([], []))
    order = []
    for r in rows:
        d = str(r[idc])
        if d not in docs:
            order.append(d)
        docs[d][0].append(r[sc])
        docs[d][1].append(r[lc] if lc else None)
    return [(d, docs[d][0], docs[d][1] if lc else None, None) for d in order]


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        return parse_obj(recs)
    if ext == ".json":
        return parse_obj(json.load(open(path, encoding="utf-8")))
    if ext in (".csv", ".tsv"):
        return parse_csv(path)
    return None


def find_label_file(data_dir):
    for dp, _, fn in os.walk(data_dir):
        for f in fn:
            if "label" in f.lower() and os.path.splitext(f)[1].lower() in (".json", ".txt"):
                p = os.path.join(dp, f)
                try:
                    if f.lower().endswith(".txt"):
                        labs = [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
                    else:
                        d = json.load(open(p, encoding="utf-8"))
                        if isinstance(d, list):
                            labs = d
                        elif isinstance(d, dict) and all(isinstance(v, int) for v in d.values()):
                            labs = list(d)
                        elif isinstance(d, dict):
                            labs = list(d.values())
                        else:
                            continue
                    if len(labs) == 13 and all(isinstance(x, str) for x in labs):
                        return p, labs
                except Exception:
                    continue
    return None, None


def inventory(data_dir, max_preview=300):
    """Human-readable listing of the dataset folder (for the notebook's inspection cell)."""
    lines = []
    for dp, _, fn in sorted(os.walk(data_dir)):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, data_dir)
            lines.append(f"{rel}  ({os.path.getsize(p) / 1024:.1f} KB)  split={split_from_path(rel)} "
                         f"domain={domain_from_text(rel)}")
            if f.lower().endswith((".json", ".jsonl", ".csv", ".tsv", ".txt")):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    head = fh.read(max_preview).replace("\n", "\\n")
                lines.append(f"    preview: {head}")
    return "\n".join(lines)


def load_phase3(data_dir, seed=42, dev_docs_per_domain_fallback=5, verbose=True, allow_dev_fallback=False):
    """-> docs: list of dict(id, split, domain, sentences, labels|None), label_file_path|None"""
    label_path, file_labels = find_label_file(data_dir)
    if file_labels is not None and set(file_labels) != LABEL_SET:
        raise Phase3FormatError(f"label file {label_path} differs from the official 13 labels: {file_labels}")

    merged = {}                                      # (split, domain, id) -> dict
    used_files = []
    for dp, _, fn in sorted(os.walk(data_dir)):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if p == label_path:
                continue
            rel = os.path.relpath(p, data_dir)
            try:
                parsed = read_any(p)
            except Phase3FormatError as e:
                raise Phase3FormatError(f"{rel}: {e}")
            except (json.JSONDecodeError, UnicodeDecodeError, csv.Error) as e:
                if verbose:
                    print(f"  skip {rel}: not parseable ({type(e).__name__})")
                continue
            if parsed is None:
                continue
            split = split_from_path(rel)
            fdom = domain_from_text(rel)
            if split is None:
                print(f"  skip {rel}: cannot tell train/dev/test from its path")
                continue
            used_files.append(rel)
            for did, sents, labs, rdom in parsed:
                dom = rdom or fdom
                if did is None:
                    raise Phase3FormatError(f"{rel}: record without a document id")
                if dom is None:
                    raise Phase3FormatError(f"{rel}: cannot tell CL/IT for doc {did} (no domain in path or record)")
                key = (split, dom, did)
                cur = merged.setdefault(key, {"id": did, "split": split, "domain": dom,
                                              "sentences": None, "labels": None, "src": []})
                cur["src"].append(rel)
                if sents is not None:
                    if cur["sentences"] is not None and cur["sentences"] != sents:
                        raise Phase3FormatError(f"doc {did}: two different sentence lists ({cur['src']})")
                    cur["sentences"] = sents
                if labs is not None:
                    if cur["labels"] is not None and cur["labels"] != labs:
                        raise Phase3FormatError(f"doc {did}: two different label lists ({cur['src']})")
                    cur["labels"] = labs
    if not merged:
        raise Phase3FormatError(f"no documents parsed under {data_dir}")

    docs = sorted(merged.values(), key=lambda d: (d["split"], d["domain"], d["id"]))
    for d in docs:
        if d["sentences"] is None:
            raise Phase3FormatError(f"{d['split']}/{d['domain']}/{d['id']}: labels found but no sentences ({d['src']})")
        d["sentences"] = ["" if s is None else str(s) for s in d["sentences"]]
        if d["split"] == "test":
            if d["labels"] is not None:
                print(f"  WARNING: test doc {d['id']} carries labels - they are DROPPED and never used")
            d["labels"] = None
            continue
        if d["labels"] is None:
            raise Phase3FormatError(f"{d['split']}/{d['domain']}/{d['id']}: no labels for a {d['split']} document")
        if len(d["labels"]) != len(d["sentences"]):
            raise Phase3FormatError(f"{d['id']}: {len(d['sentences'])} sentences vs {len(d['labels'])} labels")
        bad = [l for l in d["labels"] if l not in LABEL_SET]
        if bad:
            raise Phase3FormatError(f"{d['id']}: unknown labels {sorted(set(map(str, bad)))[:5]}")

    have = {(d["split"], d["domain"]) for d in docs}
    if not any(s == "dev" for s, _ in have):
        if not allow_dev_fallback:
            raise Phase3FormatError("no dev split found - refusing to invent one from training documents "
                                    "(pass allow_dev_fallback=True only if the dataset truly has no dev files)")
        print(f"  WARNING: no dev split found -> holding out {dev_docs_per_domain_fallback} train docs per domain")
        rng = random.Random(seed)
        for dom in DOMAINS:
            ids = sorted(d["id"] for d in docs if d["split"] == "train" and d["domain"] == dom)
            rng.shuffle(ids)
            hold = set(ids[:dev_docs_per_domain_fallback])
            for d in docs:
                if d["split"] == "train" and d["domain"] == dom and d["id"] in hold:
                    d["split"] = "dev"
    ids_by_split = defaultdict(list)
    for d in docs:
        ids_by_split[d["split"]].append(d["id"])
    for s, ids in ids_by_split.items():
        norm = [i[:-4] if i.endswith(".txt") else i for i in ids]
        dup = [i for i, c in Counter(norm).items() if c > 1]
        if dup:
            raise Phase3FormatError(f"{s}: document ids repeat across CL/IT (the scorer would reject this): {dup[:5]}")
    if verbose:
        print("  files used:", used_files)
        print("  label file:", label_path)
    return docs, label_path


def describe(docs):
    rows = []
    for split in ("train", "dev", "test"):
        for dom in DOMAINS:
            ds = [d for d in docs if d["split"] == split and d["domain"] == dom]
            n = sum(len(d["sentences"]) for d in ds)
            exp = EXPECTED_SENTENCES.get((split, dom))
            flag = "" if exp is None or exp == n else f"   <-- brief says {exp}"
            rows.append(f"  {split:<5} {dom}: {len(ds):>3} docs {n:>6} sentences{flag}")
    print("\n".join(rows))
    for split in ("train", "dev"):
        c = Counter(l for d in docs if d["split"] == split for l in d["labels"])
        print(f"  {split} label counts:", {l: c.get(l, 0) for l in LABELS})
        missing = [l for l in LABELS if c.get(l, 0) == 0]
        if missing:
            print(f"  NOTE: labels with zero {split} support: {missing}"
                  + ("  -> each scores F1=0 in official dev Macro-F1 whatever the model does" if split == "dev" else ""))
