"""Prepare a bounded, immutable FineWeb token snapshot on CPU."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from common import atomic_json, check_space, chunks, digest, identity, normalize, split_document


class NearDuplicates:
    """64-bit word SimHash; four bands find all previously seen distances <= 3.

    This is an approximate filter, not a proof that semantic duplicates are absent.
    Limit to 128 distinct words per document to bound CPU and memory costs.
    """
    def __init__(self):
        self.bands = collections.defaultdict(list)

    def duplicate(self, text):
        words = list(dict.fromkeys(re.findall(r'\w+', text.lower())))[:128]
        if not words:
            return True
        raw = b''.join(hashlib.blake2b(w.encode(), digest_size=8).digest() for w in words)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8), axis=1)
        fp = int.from_bytes(np.packbits(bits.sum(axis=0) * 2 >= len(words)).tobytes(), 'big')
        keys = [(i, (fp >> (16 * i)) & 65535) for i in range(4)]
        for key in keys:
            if any((fp ^ other).bit_count() <= 3 for other in self.bands[key]):
                return True
        for key in keys:
            self.bands[key].append(fp)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = Path(config['run_dir']); destination = run / 'dataset'
    if destination.exists():
        manifest = json.loads((destination/'manifest.json').read_text())
        if manifest['preparation_identity'] != identity(config['data']):
            raise ValueError('Existing dataset uses a different preparation recipe')
        for key in ('sequence_length', 'seed', 'model_revision'):
            if manifest[key] != config[key]:
                raise ValueError(f'Existing dataset differs in {key}')
        for name, expected in manifest['files'].items():
            if digest(destination/name) != expected:
                raise ValueError(f'Dataset hash mismatch: {name}')
        print('Dataset already prepared', flush=True); return
    check_space(Path(config['data_root']), 4_000_000_000)
    work = run / 'dataset.partial'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    # Raw document termination, distinct from an assistant turn terminator.
    eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError('Missing end-of-document token')
    files = json.loads((run/'fineweb-files.json').read_text())
    seen = set(); near = NearDuplicates(); stats = collections.Counter()
    indexes = {s: [] for s in ('train', 'validation', 'test')}
    handles = {s: (work/f'{s}.bin').open('wb') for s in indexes}
    offsets = collections.Counter(); started = time.monotonic()
    provenance = (work/'documents.jsonl').open('w')
    try:
        stop = False
        for source_path in files['paths']:
            for batch in pq.ParquetFile(source_path).iter_batches(batch_size=128, columns=['text', 'url']):
                texts, records = [], []
                for row in batch.to_pylist():
                    stats['scanned_documents'] += 1
                    text = normalize(row['text'] or '')
                    if len(text) < 256 or len(text) > 500_000:
                        stats['length_rejected'] += 1; continue
                    h = hashlib.sha256(text.casefold().encode()).hexdigest()
                    if h in seen:
                        stats['exact_duplicates'] += 1; continue
                    seen.add(h)
                    if near.duplicate(text):
                        stats['near_duplicates'] += 1; continue
                    split = split_document(h, row.get('url'), config['seed'])
                    texts.append(text); records.append((h, row.get('url'), split))
                if texts:
                    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
                    for ids, (h, url, split) in zip(encoded, records):
                        provenance.write(json.dumps({'id': h, 'url': url, 'split': split})+'\n')
                        for chunk in chunks(ids, config['sequence_length'], eos):
                            indexes[split].append((offsets[split], len(chunk)))
                            np.asarray(chunk, dtype='<u4').tofile(handles[split])
                            offsets[split] += len(chunk)
                        stats[f'{split}_documents'] += 1
                if stats['scanned_documents'] % 4096 < 128:
                    print(json.dumps({'phase':'preparing', 'tokens':dict(offsets), **stats}), flush=True)
                    check_space(Path(config['data_root']))
                if (offsets['train'] >= config['data']['max_train_tokens']
                    or stats['scanned_documents'] >= config['data']['max_documents']
                    or time.monotonic()-started > config['data']['max_prepare_seconds']):
                    stop = True; break
            if stop:
                break
    finally:
        provenance.close()
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
    for split, index in indexes.items():
        if len(index) < 16:
            raise ValueError(f'Too few examples in {split}')
        np.save(work/f'{split}.npy', np.asarray(index, dtype=np.int64))
    metadata = {
        'schema_version':1, 'preparation_identity':identity(config['data']),
        'source':files, 'model_revision':config['model_revision'],
        'tokenizer_sha256':digest(Path(config['model_path'])/'tokenizer.json'),
        'sequence_length':config['sequence_length'], 'seed':config['seed'],
        'tokens':dict(offsets), 'stats':dict(stats), 'eos_token_id':eos,
        'deduplication':'exact normalized text and approximate 64-bit word SimHash distance <=3',
        'files':{p.name:digest(p) for p in sorted(work.iterdir()) if p.is_file()},
    }
    metadata['snapshot_id'] = identity(metadata)
    atomic_json(work/'manifest.json', metadata)
    for p in work.iterdir():
        p.chmod(0o444)
    os.replace(work, destination)
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
