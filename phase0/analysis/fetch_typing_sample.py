"""Retrieve one bounded How We Type video using HTTP ranges and verify its ZIP CRC."""
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path

URL='https://zenodo.org/records/4034268/files/Reference%20video.zip?download=1'
SIZE=29292760284


def get(start,end):
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=60) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
            raise ValueError('Server did not honor exact range')
        b=r.read(end-start+2)
    if len(b)!=end-start+1:
        raise ValueError('Truncated range')
    return b


def inventory():
    tail=get(SIZE-131072,SIZE-1)
    loc=struct.unpack_from('<4sLQL',tail,tail.rfind(b'PK\x06\x07'))
    end=struct.unpack('<4sQ2H2L4Q',get(loc[2],loc[2]+55))
    size,offset=end[-2:]
    cd=get(offset,offset+size-1)
    rows=[];pos=0
    while cd[pos:pos+4]==b'PK\x01\x02':
        h=struct.unpack_from('<4s6H3L5H2L',cd,pos)
        n,x,c=h[10:13]
        uncompressed,compressed,offset=h[9],h[8],h[-1]
        extra=cd[pos+46+n:pos+46+n+x]
        at=0
        while at+4<=len(extra):
            kind,length=struct.unpack_from('<HH',extra,at)
            if kind==1:
                q=at+4
                vals=[uncompressed,compressed,offset]
                for k in range(3):
                    if vals[k]==0xffffffff:
                        vals[k]=struct.unpack_from('<Q',extra,q)[0];q+=8
                uncompressed,compressed,offset=vals
            at+=4+length
        rows.append(dict(name=cd[pos+46:pos+46+n].decode('utf-8'),size=uncompressed,
            compressed=compressed,offset=offset,method=h[4],crc=h[7]))
        pos+=46+n+x+c
    return rows


def main():
    rows=inventory()
    root=Path('.cache/nextgen/public_video');root.mkdir(parents=True,exist_ok=True)
    Path('results/nextgen/resources/hwt_video_inventory.json').write_text(json.dumps(rows,indent=2))
    entry=min((r for r in rows if r['name'].lower().endswith('.mp4') and '/._' not in r['name']),key=lambda r:r['size'])
    if entry['size']>300_000_000 or entry['compressed']>300_000_000 or entry['method']!=8:
        raise ValueError('Unexpected sample size or compression')
    target=root/Path(entry['name']).name
    if target.exists():
        print(f'Already present: {target}');return
    header=struct.unpack('<4s5H3L2H',get(entry['offset'],entry['offset']+29))
    start=entry['offset']+30+header[-2]+header[-1]
    decomp=zlib.decompressobj(-15)
    crc=0;count=0
    temp=target.with_suffix('.partial')
    with temp.open('wb') as f:
        for pos in range(0,entry['compressed'],4_000_000):
            b=get(start+pos,start+min(pos+4_000_000,entry['compressed'])-1)
            out=decomp.decompress(b,entry['size']-count+1)
            count+=len(out)
            if count>entry['size']:
                raise ValueError('Decompression limit exceeded')
            crc=zlib.crc32(out,crc);f.write(out)
            print(f'{count/1e6:.1f}/{entry["size"]/1e6:.1f} MB',flush=True)
    if count!=entry['size'] or crc!=entry['crc'] or not decomp.eof:
        raise ValueError('ZIP integrity check failed')
    temp.rename(target)
    (root/'provenance.json').write_text(json.dumps(dict(url=URL,entry=entry,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),use='non-commercial research; see source license'),indent=2))


if __name__=='__main__':
    main()
