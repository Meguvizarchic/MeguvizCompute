"""Exercise actual CLI in an exclusive output directory; preserve all results."""
from pathlib import Path
import hashlib
import json
import math
import struct
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
exe = Path(sys.argv[1]).resolve(strict=True)
provider = sys.argv[2]
run = root / 'build' / ('verify-statistics-' + provider + '-' + str(time.time_ns()))
run.parent.mkdir(parents=True, exist_ok=True)
run.mkdir(exist_ok=False)
records = []
def sha(data): return hashlib.sha256(data).hexdigest()
def write(name, content):
    p = run/name
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return p
def call(name, arguments, success):
    p=subprocess.run([str(exe),*map(str,arguments)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    write(name+'.stdout',p.stdout);write(name+'.stderr',p.stderr)
    records.append({'name':name,'arguments':list(map(str,arguments)),'exit_code':p.returncode,
                    'stdout_sha256':sha(p.stdout),'stderr_sha256':sha(p.stderr)})
    (run/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    if (p.returncode==0)!=success: raise AssertionError(name+' unexpected exit')
    return p
try:
    call('legacy',[],True)
    data=b'{"schema_version":1,"width":2,"height":2,"values":[-3,1,5,9]}\n'
    fixture=write('grid.json',data); report=run/'grid-report.json'
    call('grid',['--provider',provider,'--fixture',fixture,'--report',report],True)
    d=json.loads(report.read_text()); assert d['status']=='PASS' and d['runtime_provider']==provider and d['kernel_provider']=='cpu-reference' and d['float32_byte_order']==sys.byteorder
    assert (d['count'],d['finite_count'],d['min'],d['max'],d['sum'])==(4,4,-3,9,12)
    assert all(abs(a-b)<1e-6 for a,b in zip(d['normalized'],[0,1/3,2/3,1]))
    assert d['fixture_sha256']==sha(data) and d['input_float32_native_sha256']==sha(struct.pack('=4f',-3,1,5,9))
    assert d['output_float32_native_sha256']==sha(struct.pack('=4f',*d['normalized']))
    const=write('constant.json','{"height":1,"values":[0.25,0.25],"width":2,"schema_version":1}')
    const_report=run/'constant-report.json'; call('constant',['--provider',provider,'--fixture',const,'--report',const_report],True)
    assert json.loads(const_report.read_text())['normalized']==[0,0]
    negatives={
        'bad_shape':'{"schema_version":1,"width":2,"height":2,"values":[1]}',
        'duplicate':'{"schema_version":1,"width":1,"width":1,"height":1,"values":[0]}',
        'unknown':'{"schema_version":1,"width":1,"height":1,"values":[0],"extra":0}',
        'nonfinite':'{"schema_version":1,"width":1,"height":1,"values":[NaN]}',
        'overflow':'{"schema_version":1,"width":1,"height":1,"values":[1e100]}',
        'trailing':'{"schema_version":1,"width":1,"height":1,"values":[0]} trailing',
        'leading_zero':'{"schema_version":1,"width":1,"height":1,"values":[01]}',
        'oversized':' '* (2*1024*1024+1),
    }
    for name,text in negatives.items():
        bad=write(name+'.json',text); out=run/(name+'-report.json')
        call(name,['--provider',provider,'--fixture',bad,'--report',out],False);assert not out.exists()
    call('unknown_provider',['--provider','ncnn-vulkan','--fixture',fixture,'--report',run/'bad-provider.json'],False)
    call('duplicate_option',['--provider',provider,'--provider',provider,'--fixture',fixture,'--report',run/'duplicate-option.json'],False)
    call('missing_fixture',['--provider',provider,'--fixture',run/'not-found.json','--report',run/'missing-report.json'],False)
    before=fixture.read_bytes();call('protect_input',['--provider',provider,'--fixture',fixture,'--report',fixture],False);assert fixture.read_bytes()==before
    existing=write('existing-report.json','sentinel');call('protect_report',['--provider',provider,'--fixture',fixture,'--report',existing],False);assert existing.read_text()=='sentinel'
    sources={str(p.relative_to(root)):sha(p.read_bytes()) for p in [root/'CMakeLists.txt',root/'examples/tensor_statistics.c',
        root/'examples/tensor_statistics_cli.cpp',root/'examples/statistics_sha256.hpp',root/'tests/data/statistics_v1.json',Path(__file__)]}
    (run/'summary.json').write_text(json.dumps({'state':'PASS','scope':'CPU_GRID_CLI_HOST_ONLY','provider':provider,
        'executable_sha256':sha(exe.read_bytes()),'sources':sources,'command_count':len(records)},indent=2)+'\n')
    print(run)
except BaseException as e:
    (run/'failure.json').write_text(json.dumps({'state':'FAILED_OR_INCOMPLETE','error':repr(e)},indent=2)+'\n')
    raise
