"""Freeze only WT atomic geometry and complete WT/candidate secondary structures."""
import hashlib,json
from pathlib import Path
import RNA
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ASSETS=PROJECT/'experiments/cr_ncrna_evidence_b_v3/assets'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fold(seq):
 md=RNA.md();md.temperature=37.0
 return RNA.fold_compound(seq,md).mfe()[0]
def geometry(path,meta,wt):
 data=MMCIF2Dict(str(path));mapping={int(v['pdb_label_seq_id']):int(k) for k,v in meta['positions'].items()}
 columns=['label_asym_id','label_seq_id','label_comp_id','label_atom_id','type_symbol','Cartn_x','Cartn_y','Cartn_z','pdbx_PDB_model_num']
 residues={i:{} for i in range(1,len(wt)+1)}
 for chain,pos,base,atom,element,x,y,z,model in zip(*(data['_atom_site.'+c] for c in columns)):
  if chain!=meta['task']['chain'] or pos in ('.','?') or int(pos) not in mapping or model!='1' or element in ('H','D'):continue
  i=mapping[int(pos)]
  if base.replace('T','U')!=wt[i-1]:raise ValueError('WT coordinate sequence mismatch')
  if atom in residues[i]:raise ValueError('Ambiguous atom conformer')
  residues[i][atom]=[float(x),float(y),float(z)]
 if any(not atoms for atoms in residues.values()):raise ValueError('Missing WT coordinates')
 return {'kind':meta['task']['kind'],'units':'angstrom','indexing':'1-based WT sequence position','residues':[{'position':i,'base':wt[i-1],'atoms':atoms} for i,atoms in residues.items()]}

def main():
 output=HERE/'raw_structures.json'
 if output.exists():raise RuntimeError('Do not overwrite frozen structures')
 tasks=json.load(open(PROJECT/'experiments/cr_ncrna_evidence_b_repro_v4/tasks.json'))['tasks'];out={};audit={}
 for t in tasks:
  name=t['assay'];ctx=ASSETS/'contexts'/f'{name}.json';wt=json.load(open(ctx))['wt_sequence'].upper().replace('T','U')
  raw=ASSETS/'evidence'/name/'raw';meta_path=raw/('af3_tertiary.json' if (raw/'af3_tertiary.json').exists() else 'verified_pdb.json');meta=json.load(open(meta_path))
  cif=raw/('af3_model.cif' if (raw/'af3_model.cif').exists() else meta['task']['pdb_id']+'.cif')
  wt_dot=fold(wt);old=json.load(open(raw/'secondary.json'))
  if wt_dot!=old['task']['secondary_structure']['dot_bracket']:raise ValueError('WT fold changed')
  sample=Path(t['samples'][str(t['maximum_n'])]['path']);rows=json.load(open(sample))['variants']
  variants={f'M{i:02d}':{'dot_bracket':fold(row['seq'].upper().replace('T','U'))} for i,row in enumerate(rows,1)}
  out[name]={'wt_secondary_structure':{'dot_bracket':wt_dot},'wt_tertiary_structure':geometry(cif,meta,wt),'variant_secondary_structures':variants}
  audit[name]={'inputs':{str(p):sha(p) for p in (ctx,cif,meta_path,sample,raw/'secondary.json')},'tertiary_provenance':meta['task'],'candidate_count':len(rows),'wt_length':len(wt),'tool':'ViennaRNA','version':RNA.__version__,'temperature_c':37.0,'representation':'pseudoknot-free minimum-free-energy dot-bracket; energies not exported; WT heavy-atom coordinates only, no B-factors/confidence/contact-derived features'}
  print(name,len(rows),flush=True)
 output.write_text(json.dumps(out,separators=(',',':'),sort_keys=True)+'\n')
 (HERE/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
