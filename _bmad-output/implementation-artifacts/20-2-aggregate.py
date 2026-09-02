import csv, glob, statistics, sys, json

def load(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        for r in csv.DictReader(open(f,encoding='utf-8')):
            r['_file']=f.split('/')[-1]
            rows.append(r)
    return rows

FIELDS=['seg1_prefill_ms','seg1a_dispatch_overhead_ms','seg1b_prompt_encode_ms',
        'seg2_talker_to_first_chunk_ms','seg3_first_decode_ms','seg4_consumer_cushion_ms',
        'first_chunk_latency_ms','generation_wall_ms']

def summarise(rows,label):
    print('==',label,'n=%d'%len(rows))
    for k in FIELDS:
        v=[float(r[k]) for r in rows if r.get(k) not in (None,'')]
        if not v: continue
        print('  %-32s median=%9.1f  min=%9.1f  max=%9.1f'%(k,statistics.median(v),min(v),max(v)))

for pat,label in [(sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])]:
    summarise(load(pat),label)
