import importlib.util,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'work/stage_b/simple_structure_v5'
sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('structure_only_test',p/'structure_only.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_scheduler_never_submits_cr(monkeypatch,tmp_path):
 calls=[];progress=[]
 def unit(g,t,arm,rep,index,template,cfg,root):
  calls.append((arm,index,template));return {'status':'ok'}
 monkeypatch.setattr(m.r,'run_unit',unit)
 tasks=[{'assay':'mock','templates':{'cr':[{'user':'must not run'}],'structure_raw':[{'user':'structure1'},{'user':'structure2'}]}}]
 m.execute(object(),tasks,{},tmp_path,progress.append)
 assert calls==[('structure_raw',0,{'user':'structure1'}),('structure_raw',1,{'user':'structure2'})]
 assert len(progress)==2
