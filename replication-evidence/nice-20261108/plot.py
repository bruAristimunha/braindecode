"""Public evidence figure. Reads reviewed per-run aggregate CSV; no typed result values."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
rows = list(csv.DictReader((P / 'results.csv').open()))
target = json.loads((P / 'paper_targets.json').read_text())
tags = list(dict.fromkeys(r['tag'] for r in rows))
assert len(rows) == 100 // 2 and len(tags) == 5
for r in rows:
    assert float(r['orig_top1']) == float(r['port_top1'])
    assert float(r['orig_top5']) == float(r['port_top5'])
summary = {}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), gridspec_kw={'height_ratios': [1.1, 1]})
for col, metric in enumerate(['top1', 'top5']):
    label = 'Top-1' if metric == 'top1' else 'Top-5'
    means = [st.mean(float(r['orig_' + metric]) for r in rows if int(r['sub']) == s)
             for s in range(1, 11)]
    paper = target['per_subject'][metric]
    ax = axes[0, col]
    ax.plot(range(1, 11), paper, color='#D55E00', lw=.8, alpha=.5)
    ax.plot(range(1, 11), means, color='#0072B2', lw=.8, alpha=.5)
    ax.scatter([s-.10 for s in range(1,11)], paper, marker='D', s=28,
               color='#D55E00', zorder=3)
    ax.scatter([s+.10 for s in range(1,11)], means, marker='o', s=22,
               color='#0072B2', zorder=4)
    ax.scatter([s+.10 for s in range(1,11)], means, marker='s', s=75,
               facecolors='none', edgecolors='#0072B2', linewidths=1, zorder=5)
    ax.set(xlabel='Subject', ylabel=f'{label} accuracy (%)', xticks=range(1,11))
    ax.set_title(f'{chr(97+col)}  Per subject · five-training mean', loc='left', fontsize=10, weight='bold')
    ax.grid(axis='y', alpha=.16)
    ax = axes[1, col]
    vals = [float(r['orig_' + metric]) for r in rows if r['tag'] == 'nemar-main']
    mu = [target['ave'][metric], st.mean(vals), st.mean(means)]
    sd = [target['std_across_subjects'][metric], st.stdev(vals), st.stdev(means)]
    summary[metric] = {'paper': {'mean': mu[0], 'sd': sd[0]},
                       'one_training': {'mean': mu[1], 'sd': sd[1]},
                       'five_trainings': {'mean': mu[2], 'sd': sd[2]},
                       'per_subject_five_training': means}
    tol = target['tolerance_rel']
    ax.axhspan(mu[0]*(1-tol), mu[0]*(1+tol), color='#d9d9d9', alpha=.6, zorder=0)
    ax.axhline(mu[0], color='#555555', lw=.8, ls=':', zorder=1)
    for i in range(3):
        color = '#D55E00' if i == 0 else '#0072B2'
        ax.errorbar(i, mu[i], yerr=sd[i], fmt='D' if i == 0 else 'o',
                    color=color, capsize=4, elinewidth=1, ms=5, zorder=3)
        if i:
            ax.scatter(i, mu[i], marker='s', s=75, facecolors='none', edgecolors=color, zorder=4)
        ax.text(i, mu[i]+sd[i]+.035*(max(mu)+max(sd)), f'{mu[i]:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set(xlim=(-.5,2.5), ylim=(0, (max(m+s for m,s in zip(mu,sd)))*1.23),
           xticks=[0,1,2], xticklabels=['Paper', '(a) One training', '(b) Five trainings'],
           ylabel=f'{label} accuracy (%)')
    ax.set_title(f'{chr(99+col)}  Across subjects · mean ± SD', loc='left', fontsize=10, weight='bold')
    ax.grid(axis='y', alpha=.16)
fig.suptitle('NICE replication on NEMAR THINGS-EEG2', fontsize=15, weight='bold', x=.075, ha='left', y=.99)
handles = [Line2D([],[],marker='D',ls='none',color='#D55E00',label='Paper (arXiv v3)'),
           Line2D([],[],marker='o',ls='none',color='#0072B2',label="Authors' code"),
           Line2D([],[],marker='s',ls='none',markerfacecolor='none',color='#0072B2',label='braindecode port (exact overlap)')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51,.955), ncol=3, frameon=False)
fig.text(.075,.016,'10 subjects · 5 trainings per subject · 50 paired runs · same inputs and seeds\nError bars: SD across subjects, not confidence intervals. Gray band: ±5% of the paper mean.', fontsize=9)
fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.14,hspace=.43,wspace=.25)
for ext in ['png','svg','pdf']:
    fig.savefig(P / ('nice-replication.'+ext), dpi=300, facecolor='white')
plt.close(fig)
summary['checks'] = {'paired_runs':len(rows), 'port_vs_original_max_abs_accuracy_difference':0.0,
                     'tags': tags, 'csv_sha256':hashlib.sha256((P/'results.csv').read_bytes()).hexdigest()}
(P / 'numbers.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
print('FIGURE_OK')
