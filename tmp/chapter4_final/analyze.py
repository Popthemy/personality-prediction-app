from pathlib import Path
import sqlite3,json,math
import numpy as np
R=Path(__file__).resolve().parent
A=R.parents[1]/'pandora_personality/artifacts/run_20260917_133820'
traits=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']
codes=['lasso_baseline','lasso_baseline_gan','lasso_qlearn','lasso_qlearn_gan','lstm_baseline','lstm_baseline_gan','lstm_qlearn','lstm_qlearn_gan']
imb=json.loads((A/'imbalance_report.json').read_text())
def binary(y,p):
 tp=sum(a==b==1 for a,b in zip(y,p));tn=sum(a==b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p))
 div=lambda a,b:a/b if b else 0
 return dict(accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),specificity=div(tn,tn+fp),tp=tp,tn=tn,fp=fp,fn=fn)
models={}
for i,code in enumerate(codes,1):
 d=json.loads((A/code/'metrics.json').read_text());rows=[]
 for t in traits:
  k=t if t in d['validation']['per_trait'] else t[0]
  v=d['validation']['per_trait'][k];te=d['test']['per_trait'][k];tau=d['threshold_selection']['per_trait'][k]['threshold']
  if isinstance(v['threshold_sweep'],list):
   b=min(v['threshold_sweep'],key=lambda x:abs(x['threshold']-tau));assert abs(b['threshold']-tau)<.00011
   bm={z:b[z if z!='f1' else 'f1_score'] for z in ['accuracy','precision','recall','f1','specificity']}
  else:
   # Recover binary accuracy from binary sensitivity, specificity and known class supports.
   support=imb['splits']['validation']['traits'][t[0]];pos=support['binary_high'];neg=support['binary_low']
   tp=round(v['recall']*pos);tn=round(v['specificity']*neg)
   bm=dict(accuracy=(tp+tn)/(pos+neg),precision=v['precision'],recall=v['recall'],f1=v['f1'],specificity=v['specificity'])
  rows.append(dict(trait=t,threshold=tau,validation=v,test={k:v for k,v in te.items() if k!='threshold_sweep'},binary_validation=bm))
 models[code]=dict(id='E'+str(i),rows=rows,seconds=d['training_seconds'],comments=d['mean_comments_selected'],gan=d['targeted_gan'],validation=d['validation']['overall'],test=d['test']['overall'])
c=sqlite3.connect('file:db.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
preds=[dict(x) for x in c.execute("select * from pandora_prediction_run where status='completed' order by created_at desc,id desc limit 8")]
sets=[]
for p in preds:
 profiles=[dict(x) for x in c.execute('select * from pandora_test_profile where prediction_run_id=? order by pandora_user_id',(p['id'],))];sets.append(set(x['pandora_user_id'] for x in profiles));pr=[];all_y=[];all_p=[]
 for t in traits:
  tl=t.lower();y=np.array([x['true_'+tl] for x in profiles]);yp=np.array([x['predicted_'+tl] for x in profiles]);yb=[json.loads(x['true_binary'])[tl] for x in profiles];pb=[json.loads(x['predicted_binary'])[tl] for x in profiles];all_y+=yb;all_p+=pb
  pos=yp[np.array(yb)==1];neg=yp[np.array(yb)==0]
  auc=float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg]))
  ap=0;previous=0
  for threshold in sorted(set(yp),reverse=True):
   chosen=yp>=threshold;tp=int(np.array(yb)[chosen].sum());ap+=(tp-previous)/len(pos)*tp/int(chosen.sum());previous=tp
  b=binary(yb,pb);b.update(mae=float(abs(y-yp).mean()),rmse=float(np.sqrt(((y-yp)**2).mean())),r2=float(1-((y-yp)**2).sum()/((y-y.mean())**2).sum()),pearson=float(np.corrcoef(y,yp)[0,1]) if np.std(yp)>1e-12 else 0,roc_auc=auc,pr_auc=ap,trait=t)
  pr.append(b)
 p['traits']=pr;p['pooled']=binary(all_y,all_p);p['means']={k:sum(x[k] for x in pr)/5 for k in ['mae','rmse','r2','pearson','roc_auc','pr_auc','accuracy','precision','recall','f1','specificity']};p['thresholds']=json.loads(profiles[0]['metrics'])['decision_thresholds'];p['comments']=sum(x['comment_count'] for x in profiles)
 assert all(abs(p[k]-p['pooled'][k if k!='f1_score' else 'f1'])<1e-9 for k in ['accuracy','precision','recall','f1_score','specificity'])
 print(p['id'],models[p['condition']]['id'],p['predicted_samples'],{k:round(v,4) for k,v in p['means'].items()})
data=json.loads((A/'experiment_data.json').read_text());print('data keys',list(data))
repro=json.loads((A/'reproducibility.json').read_text());splits=repro['split'];overlap={k:len(sets[0]&set(v)) for k,v in splits.items() if isinstance(v,list)}
print('overlap',overlap,'same33',all(s==sets[0] for s in sets if len(s)==33),'subset20',all(s<=sets[0] for s in sets))
out=dict(models=models,predictions=preds,overlap=overlap,quality=json.loads((A/'data_quality.json').read_text()),imbalance=imb)
(R/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8')
