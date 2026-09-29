"""Local synchronous newline-JSON RPC server for the excavator controller.

Run in the source Python environment, separate from Isaac's bundled Python:
    python bridge_server.py --port 47653
Only loopback is bound. A single client owns a single ordered simulation. No
remote commands, dynamic expressions or arbitrary module execution are exposed.
"""
from __future__ import annotations
import argparse
import json
import socket
import sys
import traceback
from bridge import ControllerBridge, jsonable

MAX_REQUEST = 8 * 1024 * 1024


def serve(port=47653):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',int(port))); listener.listen(1)
        print(f'EXCAVATOR_BRIDGE_READY 127.0.0.1:{listener.getsockname()[1]}',flush=True)
        conn,address=listener.accept()
        with conn,conn.makefile('rwb') as stream:
            bridge=None
            while True:
                raw=stream.readline(MAX_REQUEST+1)
                if not raw:break
                if len(raw)>MAX_REQUEST or not raw.endswith(b'\n'):
                    raise ValueError('Oversized or unterminated request')
                close=False
                try:
                    request=json.loads(raw)
                    op=request.pop('op',None)
                    if op=='initialize':
                        if bridge is not None:raise ValueError('Already initialized')
                        bridge=ControllerBridge(**request);result=bridge.metadata()
                    elif op=='tick':
                        if bridge is None:raise ValueError('Initialize first')
                        result=bridge.tick(**request)
                    elif op=='finalize':
                        if bridge is None:raise ValueError('Initialize first')
                        result=bridge.finalize(**request)
                    elif op=='summary':
                        result=bridge.summary() if bridge is not None else {'initialized':False}
                    elif op=='shutdown':
                        result=bridge.summary() if bridge is not None else {'initialized':False};close=True
                    else:raise ValueError(f'Unsupported operation: {op}')
                    response={'ok':True,'result':jsonable(result)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response={'ok':False,'error':str(error),'exception_type':type(error).__name__}
                    # Failed control transactions cannot safely resume.
                    close=True
                stream.write((json.dumps(response,allow_nan=False,separators=(',',':'))+'\n').encode());stream.flush()
                if close:break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=47653)
    args=parser.parse_args();serve(args.port)
