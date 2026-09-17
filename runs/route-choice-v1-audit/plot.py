"""Plot matched flight paths for the four selected SA route-choice probes."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
runs=root.parent
results=json.loads((root/'targeted-results.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,11))
for ax,episode in zip(axes.flat,(179,181,86,96)):
    scenario=json.loads((runs/f'diagnose-residual-v1-sa-episode{episode}'/'scenario.json').read_text())
    center=np.asarray(scenario['center'])
    def xy(points):
        p=np.asarray(points)
        return (p-center)[:,::-1]*[60*1.852*np.cos(np.deg2rad(center[0])),60*1.852]
    sector=xy(scenario['sector'])
    ax.add_patch(Polygon(sector,closed=True,fill=False,edgecolor='#475569',linewidth=1.2))
    for obstacle in scenario['obstacles']:
        ax.add_patch(Polygon(xy(obstacle['vertices']),closed=True,facecolor='#f4c7c3',edgecolor='#bf7069',linewidth=.5))
    for variant,color,style in [('original','#7c3aed','--'),('choice','#007f8b','-')]:
        with (root/f'sa{episode}-{variant}.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        points=xy([(float(r['lat']),float(r['lon'])) for r in rows])
        ax.plot(*points.T,color=color,linestyle=style,linewidth=1.6)
        ax.scatter(*points[-1],color=color,s=24,zorder=4)
    spec=scenario['agents'][0]
    start,goal=xy([spec['start'],spec['goal']])
    ax.scatter(*start,color='#1e293b',marker='o',s=40,zorder=5)
    ax.scatter(*goal,color='#1e293b',marker='*',s=120,zorder=5)
    case=results['cases'][str(episode)]
    old,new=case['original']['metrics'],case['route_choice']['metrics']
    state=lambda m:'arrival' if m['waypoint_reached'] else 'timeout'
    ax.set_title(f"Scenario {episode}: {state(old)} {old['flight_time']:.0f} s → {state(new)} {new['flight_time']:.0f} s",fontsize=11,loc='left',pad=25)
    ax.text(0,1.02,f"Intrusion: {old['intrusion_time']:.0f} → {new['intrusion_time']:.0f} s   |   Restricted: {old['time_in_restricted_area']:.0f} → {new['time_in_restricted_area']:.0f} s",transform=ax.transAxes,fontsize=9,color='#475569')
    extent=np.ptp(sector,axis=0)*.06
    ax.set_xlim(sector[:,0].min()-extent[0],sector[:,0].max()+extent[0])
    ax.set_ylim(sector[:,1].min()-extent[1],sector[:,1].max()+extent[1])
    ax.set_aspect('equal');ax.set_xlabel('East (km)');ax.set_ylabel('North (km)');ax.grid(alpha=.15)
fig.suptitle('Shorter routes recover arrivals, with a conflict tradeoff',fontsize=17,y=.985)
fig.text(.5,.95,'Identical learned weights • Four selected development failures • Original one-second scoring',ha='center',fontsize=10,color='#475569')
handles=[Line2D([0],[0],color='#7c3aed',linestyle='--',label='Original routing'),Line2D([0],[0],color='#007f8b',label='Route choice'),Line2D([0],[0],color='#1e293b',marker='o',linestyle='',label='Start'),Line2D([0],[0],color='#1e293b',marker='*',linestyle='',label='Goal')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.028),ncol=4,frameon=False)
fig.text(.5,.009,'Paths sampled every 10 seconds. Intruder paths omitted for clarity. Scenario indices are zero based; this is not a population estimate.',ha='center',fontsize=8,color='#475569')
fig.subplots_adjust(left=.07,right=.97,top=.89,bottom=.115,hspace=.35,wspace=.24)
fig.savefig(root/'targeted-paths.png',dpi=150)
fig.savefig(root/'targeted-paths.pdf')
plt.close(fig)
print(root/'targeted-paths.png')
