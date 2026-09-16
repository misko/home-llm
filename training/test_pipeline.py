import json
from pathlib import Path
import numpy as np
import pytest
from common import chunks, split_document, atomic_json, digest, verify_checkpoint
from prepare import NearDuplicates


def test_raw_document_chunks_terminate_once():
    result=list(chunks([10,11,12,13,14],3,99))
    assert result==[[10,11,12],[13,14,99]]
    assert sum(x.count(99) for x in result)==1


def test_qwen_vocabulary_is_preserved(tmp_path):
    p=tmp_path/'tokens.bin'
    np.asarray([248044,248320-1],dtype='<u4').tofile(p)
    assert np.fromfile(p,dtype='<u4').tolist()==[248044,248319]


def test_duplicate_and_url_grouping():
    assert split_document('a','https://EXAMPLE.org/page?q=1',42)==split_document('b','https://example.org/page?q=2',42)
    assert split_document('a',None,42)==split_document('a',None,42)
    near=NearDuplicates()
    text=' '.join('token'+str(i) for i in range(100))
    assert not near.duplicate(text)
    assert near.duplicate(text+' token100')


def test_checkpoint_corruption_detected(tmp_path):
    p=tmp_path/'adapter.safetensors';p.write_bytes(b'weights')
    atomic_json(tmp_path/'manifest.json',{'files':{p.name:digest(p)}})
    verify_checkpoint(tmp_path)
    p.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        verify_checkpoint(tmp_path)

