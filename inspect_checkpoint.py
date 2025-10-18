import sys
import torch
import os

def inspect(path):
    print('Inspecting checkpoint:', path)
    if not os.path.exists(path):
        print('File not found')
        return
    ck = torch.load(path, map_location='cpu')
    print('Top-level type:', type(ck))
    if isinstance(ck, dict):
        keys = list(ck.keys())
        print(f'Top-level keys ({len(keys)}):', keys[:100])
        # try some common wrapper keys
        candidates = ['state_dict','model_state','model','model_state_dict']
        found = False
        for cand in candidates:
            if cand in ck:
                found = True
                st = ck[cand]
                print(f"Found nested key '{cand}' with type {type(st)} and {len(st) if isinstance(st, dict) else 'N/A'} keys")
                try:
                    sample_keys = list(st.keys())[:200]
                    print('Sample nested keys:', sample_keys)
                    # print shapes/types for a few
                    cnt=0
                    for k in sample_keys:
                        v = st[k]
                        t = type(v)
                        s = ''
                        try:
                            s = getattr(v, 'shape', None)
                        except Exception:
                            s = None
                        print(f'  {k} -> type={t}, shape={s}')
                        cnt+=1
                        if cnt>=20:
                            break
                except Exception as e:
                    print('Could not inspect nested state:', e)
                break
        if not found:
            # maybe ck itself is state-dict-like
            try:
                # check a few keys' types
                sample = keys[:200]
                print('Assuming top-level is a state-dict; sample keys:', sample)
                cnt=0
                for k in sample:
                    v = ck[k]
                    t = type(v)
                    s = None
                    try:
                        s = getattr(v, 'shape', None)
                    except Exception:
                        s = None
                    print(f'  {k} -> type={t}, shape={s}')
                    cnt+=1
                    if cnt>=20:
                        break
            except Exception as e:
                print('Could not introspect top-level dict:', e)
    else:
        print('Checkpoint is not a dict. Type:', type(ck))

if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: python inspect_checkpoint.py <path-to-pth>')
    else:
        inspect(sys.argv[1])
