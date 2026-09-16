"""Measure a private client's real CNN forward pass, without SGD or mesh work."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hudes-root',type=Path,default=Path('/home/mouse9911/gits/human_descent'))
parser.add_argument('--url',default='ws://127.0.0.1:10000')
parser.add_argument('--samples',type=int,default=3)
parser.add_argument('--interval',type=float,default=30)
args=parser.parse_args()
sys.path.insert(0,str(args.hudes_root))
from hudes import hudes_pb2
from websockets.asyncio.client import connect


async def main():
    async with connect(args.url,max_size=8*1024*1024,open_timeout=10) as socket:
        config=hudes_pb2.Control(type=hudes_pb2.Control.CONTROL_CONFIG,config=hudes_pb2.Config(
            seed=20260913,dims_at_a_time=1,mesh_grid_size=3,mesh_step_size=.01,mesh_grids=0,
            batch_size=32,dtype='float32',mesh_enabled=False,loss_lines=0,resume_supported=False))
        await socket.send(config.SerializeToString())
        async def response(index=None):
            while True:
                raw=await asyncio.wait_for(socket.recv(),timeout=30)
                msg=hudes_pb2.Control();msg.ParseFromString(raw)
                if msg.type==hudes_pb2.Control.CONTROL_TRAIN_LOSS_AND_PREDS and (index is None or msg.request_idx==index):
                    return msg
        await response()  # Exclude private-client initialization/warmup.
        index=0
        while args.samples==0 or index<args.samples:
            index+=1
            msg=hudes_pb2.Control(type=hudes_pb2.Control.CONTROL_DIMS,request_idx=index,
                dims_and_steps=[hudes_pb2.DimAndStep(dim=0,step=0.0)])
            started=time.monotonic()
            await socket.send(msg.SerializeToString())
            result=await response(index)
            print(json.dumps({'timestamp':time.time(),'latency_seconds':time.monotonic()-started,
                              'task':'cnn3-forward-batch32-zero-step','request_idx':index,
                              'loss':result.train_loss_and_preds.train_loss}),flush=True)
            if args.samples==0 or index<args.samples:await asyncio.sleep(args.interval)


try:
    asyncio.run(main())
except Exception as exc:
    print(json.dumps({'error':f'{type(exc).__name__}: {exc}','timestamp':time.time()}),flush=True)
    raise
