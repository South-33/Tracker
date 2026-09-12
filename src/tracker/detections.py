"""Deterministic frozen top-K detector outputs cached for causal research rollouts."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch

from .features import FrozenFeatures
from .proposals import decode_proposals, top_candidates, DetectionPrediction


class FrozenDetections:
    def __init__(self, base, size, device, feature_directory, directory,
                 candidate_count=96, cache_mib=512):
        self.base=base; self.device=device; self.candidate_count=int(candidate_count)
        self.features=FrozenFeatures(base,size,device,feature_directory,0)
        self.limit=max(0,int(cache_mib*2**20)); self.items=OrderedDict(); self.bytes=0
        self.hits=self.disk_hits=self.decodes=0
        digest=hashlib.sha256()
        settings={"size":int(size),"candidate_count":self.candidate_count,"torch":torch.__version__,
                  "precision":"cuda_amp_fp16" if str(device).startswith("cuda") else "cpu_fp32",
                  "decoder":"clean-rtdetr-topk-v1"}
        digest.update(json.dumps(settings,sort_keys=True).encode())
        for key,value in base.detector.state_dict().items():
            digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.directory=Path(directory)/digest.hexdigest()[:24]
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')

    def get(self, sequence, number):
        image_path=sequence.directory/'img1'/f'{number:08d}.jpg'
        stat=image_path.stat()
        key=f'{sequence.name}-{number}-{stat.st_size}-{stat.st_mtime_ns}-{sequence.record["gt_sha256"][:16]}'
        if key in self.items:
            self.hits+=1; self.items.move_to_end(key); return self.items[key][0]
        path=self.directory/f'{key}.pt'
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=True); self.disk_hits+=1
        else:
            pyramid,ids,target=self.features.get(sequence,number)
            with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.float16,
                enabled=str(self.device).startswith('cuda')):
                full=decode_proposals(self.base,pyramid)
                detection,indices=top_candidates(full,self.candidate_count)
            payload={
                'boxes':detection.boxes.detach().cpu().half(),
                'person_logits':detection.person_logits.detach().cpu().half(),
                'features':detection.features.detach().cpu().half(),
                'valid':detection.valid.detach().cpu(),
                'indices':indices.detach().cpu(),
                'ids':torch.from_numpy(ids.copy()), 'target':target.detach().cpu(),
            }
            tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path); self.decodes+=1
        detection=DetectionPrediction(payload['boxes'].to(self.device),payload['person_logits'].to(self.device),
                                      payload['features'].to(self.device),payload['valid'].to(self.device))
        result=(detection,payload['indices'].to(self.device),payload['ids'].numpy(),payload['target'].to(self.device))
        size=sum(x.numel()*x.element_size() for x in detection)+payload['indices'].numel()*payload['indices'].element_size()+payload['target'].numel()*payload['target'].element_size()+payload['ids'].numel()*payload['ids'].element_size()
        if size<=self.limit:
            while self.bytes+size>self.limit and self.items:
                _,(_,removed)=self.items.popitem(last=False); self.bytes-=removed
            self.items[key]=(result,size); self.bytes+=size
        return result

    def stats(self):
        return {"memory_hits":self.hits,"disk_hits":self.disk_hits,"fresh_decodes":self.decodes,
                "resident_mib":self.bytes/2**20,"limit_mib":self.limit/2**20,"directory":str(self.directory),
                "feature_cache":self.features.stats()}
