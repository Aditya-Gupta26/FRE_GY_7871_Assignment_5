"""Build analysis.ipynb: the report's sections (verbatim from REPORT.md) + code cells that
recompute and show every table and figure. Then run:
    jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
"""
import _bootstrap  # noqa: F401
import re

import nbformat as nbf

from notebook_text import MD
from src.config import ROOT

report = (ROOT / "REPORT.md").read_text()
parts = re.split(r"(?m)^(?=## )", report)
head, sections = parts[0], {re.match(r"## (\d+)", p).group(1): p for p in parts[1:]}

C = {}
C["setup"] = """import sys, json, warnings, subprocess
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 180); pd.set_option('display.max_columns', 30)
from src.config import TABLES, FIGURES, OUTPUTS, PREREG, PREREG_DEVIATION
# recompute every table and figure from the saved data, exactly as for the report
for s in ['scripts/08_run_analysis.py', 'scripts/09_make_figures.py']:
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print(s, 'exit', r.returncode)
    assert r.returncode == 0, r.stderr[-2000:]
R = json.loads((OUTPUTS / 'results.json').read_text())
print('key topics:', R['q1']['key_topics'])"""
C["data"] = """display(pd.read_csv(TABLES / 'filter_waterfall.csv'))
print(json.load(open(TABLES / 'news_corpus_counts.json')))
print('docs used in Q1:', R['q1']['n_docs'])"""
C["method"] = """print('pre-registration (written before any test):'); print(json.dumps(PREREG, indent=1))
print('disclosed deviation:'); print(json.dumps(PREREG_DEVIATION, indent=1))
print('topic model:', R['q1']['bertopic']); print('diagnostics:', json.dumps(R['q1']['diagnostics']['ari']))
display(pd.read_csv(TABLES / 'q1_lda_sweep.csv').round(4))
print('topic assignment vs blind issue labels:', R['q1']['topic_assignment_accuracy'])"""
C["validation"] = """v = pd.read_csv(TABLES / 'q2_validation.csv')
display(v.round(3))
print(json.dumps(R['q2']['validation_summary'], indent=1))"""
C["q1"] = """# topic table: every BERTopic topic with its LLM label, top words, share of all docs and one example
from src.config import LABELS
tt = pd.read_csv(LABELS / 'topic_taxonomy.csv').merge(pd.read_csv(LABELS / 'topic_sheet.csv'), on='topic')
tt['share_all_docs_%'] = (100 * tt['n_docs'] / len(pd.read_parquet('data/interim/doc_topics.parquet'))).round(2)
pd.set_option('display.max_colwidth', 120)
display(tt[['topic', 'label', 'short_name', 'confidence', 'share_all_docs_%', 'top_words', 'example_1']].sort_values(['label', 'share_all_docs_%'], ascending=[True, False]))
display((pd.read_csv(TABLES / 'q1_issue_shares.csv', index_col=0) * 100).round(1))
display(pd.read_csv(TABLES / 'q1_poll_compare.csv').round(2))
display((pd.read_csv(TABLES / 'q1_seed_stability.csv', index_col=0) * 100).round(1))
for f in ['fig1_issue_shares.png', 'figA_salience_over_time.png']:
    display(Image(filename=str(FIGURES / f), width=950))"""
C["q2"] = """print('docs in index:', R['q2']['docs_in_index'], '| median docs/day:', R['q2']['docs_per_day_median'])
print('coverage (share of days with >= 5 docs):', R['q2']['coverage_ge5'])
print('split-half reliability:', R['q2']['split_half'])
print('correlation of C with its alternatives:', R['q2']['corr_C_with_alternatives'])
display(Image(filename=str(FIGURES / 'fig2_directional_index.png'), width=950))"""
C["q3"] = """print('stationarity:', json.dumps(R['q3']['stationarity'], indent=1))
display(pd.read_csv(TABLES / 'q3_family_a.csv').round(4))
print('claim checks:', R['q3']['claims'])
g = pd.read_csv(TABLES / 'q3_robustness_grid.csv'); display(g[g.p_hac < 0.10].round(4))
print('robust p<0.10:', R['q3']['robust_count_p_lt_0.10'], 'of', R['q3']['robust_tests'], '| detectable r:', round(R['q3']['mde_r_N'], 3),
      '| venue daily corr:', round(R['q3']['venue_daily_corr'], 3), '| venue gap:', R['q3']['venue_gap_pp'])
print('VAR impulse response of dP to an index shock:', [round(x, 3) for x in R['q3']['var']['irf_dP_to_sent_shock']])
display(Image(filename=str(FIGURES / 'fig3_sweep_index.png'), width=950))"""
C["q4"] = """display(pd.read_csv(TABLES / 'q4_election_betas_main_poly.csv', index_col=0)[['group', 'expected_sign', 'beta', 'gamma_coef', 'gamma_se', 'gamma_ci90_lo', 'gamma_ci90_hi', 'sign_match']].round(3))
display(pd.read_csv(TABLES / 'q4_group_gamma_main_poly.csv', index_col=0).round(3))
display(pd.read_csv(TABLES / 'q4_basket_gamma_main_poly.csv', index_col=0).round(3))
v = pd.read_csv(TABLES / 'q4_validation_vs_dp.csv'); display(v[v.leg == 'ls'].round(3))
display(pd.read_csv(TABLES / 'q4_rigobon_sack.csv', index_col=0).round(3))
display(pd.read_csv(TABLES / 'q4_family_sent_vs_basket.csv').round(4))
display(pd.read_csv(TABLES / 'q4_per_name.csv').round(3))
print('cumulative AR at Oct 7 (%):', R['q4']['cum_ar_end'], '| provisional Oct 7 close:', R['q4']['provisional_oct7'])
display(Image(filename=str(FIGURES / 'fig4_basket.png'), width=950))"""
C["extras"] = """display(pd.read_csv(TABLES / 'e1_agenda_distance.csv').round(3))
display(pd.read_csv(TABLES / 'e2_issue_vs_sweep.csv').round(3))
display((pd.read_csv(TABLES / 'e3_issue_shift.csv', index_col=0)).round(4))
print({k: v for k, v in R['extras']['e3'].items() if k != 'xcorr_event'})"""

nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(MD["intro"]), nbf.v4.new_code_cell(C["setup"]),
         nbf.v4.new_markdown_cell(head.strip())]
plan = [("1", []), ("2", ["data"]), ("3", ["method", ("md", "score_where"), "validation"]),
        ("4", ["q1", ("md", "how_q1")]), ("5", ["q2", ("md", "how_q2")]), ("6", ["q3", ("md", "how_q3")]),
        ("7", ["q4", ("md", "how_q4")]), ("8", []), ("9", ["extras"]), ("10", [])]
for sec, extra in plan:
    cells.append(nbf.v4.new_markdown_cell(sections[sec].strip()))
    for e in extra:
        cells.append(nbf.v4.new_markdown_cell(MD[e[1]]) if isinstance(e, tuple) else nbf.v4.new_code_cell(C[e]))
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "analysis.ipynb")
print("wrote analysis.ipynb with", len(cells), "cells")
