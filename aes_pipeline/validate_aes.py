#!/usr/bin/env python3
"""
validate_aes.py
Compares the pipeline's machine scores (from a run's texts.csv) against the
clean, source-verified expert marks (AES_Clean_Dataset.xlsx), and writes a
per-criterion accuracy table.

WHY THIS EXISTS
The earlier accuracy figures were computed against scrambled labels. This script
recomputes them against the corrected gold set, so each criterion can be judged
on its own merits and you can see exactly where (if anywhere) work is needed.

IMPORTANT: the gold marks come ONLY from the clean dataset (--gold). The score
columns inside the run output are your expert marks echoed back, NOT the
machine's verdicts, so this script never reads them as "machine".

USAGE
  python validate_aes.py --run texts.csv --gold AES_Clean_Dataset.xlsx --out validation_report.xlsx

ONE THING TO SET FIRST
Tell the script which column in the run output holds the machine's FINAL score
for each criterion. Edit the PRED dictionary below. If your pipeline writes a
final integer per criterion (recommended), point each entry at that column.
Where it does not, the defaults derive a score from the band/category columns;
those derivations are marked PROVISIONAL and should be checked against your own
ladder definitions.
"""
import argparse, sys
import pandas as pd, numpy as np

CRITERIA = ['AU','TS','ID','CS/PD','Voc','Coh','Pa','SS','Pun','Spell']
MAXSCORE = {'AU':6,'TS':4,'ID':5,'CS/PD':4,'Voc':5,'Coh':4,'Pa':2,'SS':6,'Pun':5,'Spell':6}

# ----------------------------------------------------------------------------
# HOW TO READ THE MACHINE SCORE FOR EACH CRITERION FROM THE RUN OUTPUT.
# Each entry is one of:
#   ('col', 'ColumnName')                          use that column directly as the score
#   ('map', 'ColumnName', {category: score, ...})  map a text category to a score  (PROVISIONAL)
#   ('gate','BandCol','GateFlagCol')               use band, +1 if the gate flag is truthy
# Set the '<SET ME>' entries to your pipeline's final-score columns, then rerun.
# ----------------------------------------------------------------------------
PRED = {
    'AU':    ('col', 'AudienceBand'),
    'TS':    ('map', 'TextStructure', {'none':0,'minimal':1,'weak':2,'present':3,'effective':4}),   # PROVISIONAL
    'ID':    ('map', 'Ideas', {'none':0,'minimal':1,'simple':2,'substantial':3,'coherent':4,'crafted':5}), # PROVISIONAL
    'CS/PD': ('fn',  'cspd'),
    'Voc':   ('gate','Voc_Band','Voc_Band5Candidate'),
    'Coh':   ('gate','Coh_Band','Coh_Band4Candidate'),
    'Pa':    ('col', 'Pa'),
    'SS':    ('fn',  'ss'),
    'Pun':   ('fn',  'pun'),
    'Spell': ('fn',  'spell'),
}

_CAT = {'none':0,'named':1,'suggestion':2,'emerges':3,'effective':4}

def _fn_cspd(row):
    ca = _CAT.get(str(row.get('CharacterAnalysis','')).lower().strip(), None)
    sa = _CAT.get(str(row.get('SettingAnalysis','')).lower().strip(), None)
    if ca is None and sa is None: return None   # TB2 failed for this script
    ca = ca or 0; sa = sa or 0
    return max(ca, sa)

def _fn_ss(row):
    cnt = float(row.get('SS_AssessableCount') or 0)
    if cnt == 0: return 0
    pct  = float(row.get('SS_CorrectPct') or 0)
    types = int(row.get('SS_DistinctTypesUsed') or 0)
    cx_c  = int(row.get('SS_Complex_Correct') or 0)
    proj  = int(row.get('SS_ProjectedCount') or 0)
    # Cap by sentence count — can't demonstrate variety on tiny scripts
    if cnt <= 3:  cap = 2
    elif cnt <= 6:  cap = 3
    elif cnt <= 15: cap = 4
    elif cnt <= 30: cap = 5
    else:           cap = 6
    if pct < 30: s = 1
    elif pct < 65: s = 2
    elif pct < 82: s = 3
    elif pct < 93: s = 3 if types <= 1 else 4
    elif cx_c == 0 and proj == 0: s = 4
    elif pct < 98: s = 4 if types <= 2 else 5
    else: s = 5 if types <= 2 else 6
    return min(s, cap)

def _fn_pun(row):
    total = float(row.get('Pun_Sentence_Total') or 0)
    if total == 0: return 0
    sp = float(row.get('Pun_Sentence_CorrectPct') or 0)
    ot = float(row.get('Pun_Other_Total') or 0)
    op = float(row.get('Pun_Other_CorrectPct') or 100) if ot > 0 else 100
    if sp < 8:  return 0 if total >= 4 else 1
    if sp < 80: return 2
    if sp < 92: return 2 if ot == 0 else 3
    if ot == 0: return 3
    if op < 65: return 3
    if op < 88: return 4
    return 5

def _fn_spell(row):
    sc = float(row.get('Spell_Simple_Correct') or 0); si = float(row.get('Spell_Simple_Incorrect') or 0)
    cc = float(row.get('Spell_Common_Correct') or 0);  ci = float(row.get('Spell_Common_Incorrect') or 0)
    dc = float(row.get('Spell_Difficult_Correct') or 0); di = float(row.get('Spell_Difficult_Incorrect') or 0)
    xc = float(row.get('Spell_Challenging_Correct') or 0); xi = float(row.get('Spell_Challenging_Incorrect') or 0)
    total = sc+si+cc+ci+dc+di+xc+xi
    if total == 0: return 0
    sp = 100*sc/(sc+si) if (sc+si) > 0 else 100
    cp = 100*cc/(cc+ci) if (cc+ci) > 0 else 100
    dp = 100*dc/(dc+di) if (dc+di) > 0 else 100
    xp = 100*xc/(xc+xi) if (xc+xi) > 0 else 100
    xa = (xc+xi) > 0
    if sp < 50: return 0
    if cp < 60 or sp < 75: return 1
    if dp < 50 or cp < 80: return 2
    if dp < 75: return 3
    if not xa: return 4
    if xp < 75: return 4
    return 5 if dp < 95 else 6

_FN_MAP = {'cspd': _fn_cspd, 'ss': _fn_ss, 'pun': _fn_pun, 'spell': _fn_spell}

def truthy(v):
    if pd.isna(v): return False
    if isinstance(v,(int,float)): return v != 0
    return str(v).strip().lower() in ('1','true','yes','y','t')

def machine_score(row, spec):
    kind = spec[0]
    if kind == 'col':
        col = spec[1]
        if col not in row or pd.isna(row[col]): return None
        try: return int(round(float(row[col])))
        except: return None
    if kind == 'map':
        col, table = spec[1], spec[2]
        if col not in row or pd.isna(row[col]): return None
        return table.get(str(row[col]).strip().lower())
    if kind == 'gate':
        band_col, gate_col = spec[1], spec[2]
        if band_col not in row or pd.isna(row[band_col]): return None
        b = int(round(float(row[band_col])))
        if gate_col in row and truthy(row[gate_col]): b += 1
        return b
    if kind == 'fn':
        fn = _FN_MAP.get(spec[1])
        if fn is None: return None
        result = fn(row)
        return result   # may be None if TB2 failed
    return None

def qwk(y_true, y_pred, max_score):
    # quadratic weighted kappa, the standard agreement measure for ordinal AES scores
    y_true = np.asarray(y_true,int); y_pred = np.asarray(y_pred,int)
    N = max_score + 1
    O = np.zeros((N,N))
    for a,b in zip(y_true,y_pred): O[a,b]+=1
    if O.sum()==0: return np.nan
    w = np.zeros((N,N))
    for i in range(N):
        for j in range(N): w[i,j] = (i-j)**2/((N-1)**2)
    act = O.sum(1); pred = O.sum(0)
    E = np.outer(act,pred)/O.sum()
    denom = (w*E).sum()
    return 1 - (w*O).sum()/denom if denom else np.nan

def spearman(a,b):
    a=pd.Series(a); b=pd.Series(b)
    return a.corr(b, method='spearman')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='pipeline output texts.csv')
    ap.add_argument('--gold', required=True, help='AES_Clean_Dataset.xlsx')
    ap.add_argument('--gold_sheet', default='clean_dataset')
    ap.add_argument('--out', default='validation_report.xlsx')
    a = ap.parse_args()

    run = pd.read_csv(a.run)
    gold = pd.read_excel(a.gold, sheet_name=a.gold_sheet)
    run_key  = 'Research ID' if 'Research ID' in run.columns else run.columns[0]
    gold_key = 'ScriptID'    if 'ScriptID'   in gold.columns else gold.columns[0]
    run = run.set_index(run_key)
    gold = gold.set_index(gold_key)
    ids = [i for i in run.index if i in gold.index]
    print(f'Matched {len(ids)} scripts between run and gold.')

    rows, dis = [], []
    skipped = []
    for crit in CRITERIA:
        spec = PRED[crit]
        # detect unset / missing machine column
        needed = spec[1]
        if spec[0] in ('col','map') and needed not in run.columns:
            skipped.append((crit, needed)); continue
        if spec[0]=='gate' and spec[1] not in run.columns:
            skipped.append((crit, spec[1])); continue
        if spec[0]=='fn' and spec[1] not in _FN_MAP:
            skipped.append((crit, f'unknown fn: {spec[1]}')); continue
        ms, es = [], []
        for sid in ids:
            m = machine_score(run.loc[sid], spec)
            e = gold.loc[sid, crit]
            if m is None or pd.isna(e): continue
            m = max(0, min(MAXSCORE[crit], int(m)))   # clamp into valid range
            e = int(e)
            ms.append(m); es.append(e)
            if abs(m-e) >= 2:
                dis.append({'ScriptID':sid,'criterion':crit,'machine':m,'expert':e,'diff':m-e})
        if not ms:
            skipped.append((crit,'no comparable values')); continue
        ms, es = np.array(ms), np.array(es)
        exact = (ms==es).mean()
        within1 = (np.abs(ms-es)<=1).mean()
        rows.append({
            'criterion':crit, 'n':len(ms),
            'exact_%':round(100*exact,1),
            'within1_%':round(100*within1,1),
            'QWK':round(qwk(es,ms,MAXSCORE[crit]),3),
            'Spearman':round(spearman(es,ms),3),
            'bias_machine_minus_expert':round((ms-es).mean(),2),
            'machine_mean':round(ms.mean(),2),'expert_mean':round(es.mean(),2),
            'source_column':spec[1],
            'provisional':'YES (verify mapping)' if spec[0]=='map' or str(spec[1]).startswith('<SET') else ''
        })

    per = pd.DataFrame(rows)
    overall = pd.DataFrame([{
        'criteria_scored':len(per),
        'mean_exact_%':round(per['exact_%'].mean(),1) if len(per) else None,
        'mean_within1_%':round(per['within1_%'].mean(),1) if len(per) else None,
        'mean_QWK':round(per['QWK'].mean(),3) if len(per) else None,
    }])
    with pd.ExcelWriter(a.out) as w:
        per.to_excel(w, sheet_name='per_criterion', index=False)
        overall.to_excel(w, sheet_name='overall', index=False)
        pd.DataFrame(dis).to_excel(w, sheet_name='disagreements_2plus', index=False)
        if skipped:
            pd.DataFrame(skipped, columns=['criterion','missing_or_reason']).to_excel(w, sheet_name='not_scored', index=False)

    print('\nPER-CRITERION (vs clean labels):')
    if len(per):
        print(per[['criterion','n','exact_%','within1_%','QWK','Spearman','bias_machine_minus_expert','provisional']].to_string(index=False))
    if skipped:
        print('\nNOT SCORED (set the machine-score column for these in PRED):')
        for c,why in skipped: print(f'  {c}: {why}')
    print(f'\nWrote {a.out}')
    print('\nReading guide: QWK above ~0.7 is strong, 0.6-0.7 acceptable, below 0.6 needs attention.')
    print('A large positive bias means the machine over-scores that criterion; negative means under-scores.')

if __name__ == '__main__':
    main()
