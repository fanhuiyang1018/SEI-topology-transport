                      

                       

"""Plot thermodynamic time series from GPUMD thermo.out."""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



                              

df = pd.read_csv('thermo.out', sep=r'\s+', header=None,

                 usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])                                  



                               

amu2g, ang3cm3 = 1.660539e-24, 1e-24

ATOMIC_MASS = {

    'H':1.008,'He':4.003,'Li':6.941,'Be':9.012,'B':10.81,'C':12.011,

    'N':14.007,'O':15.999,'F':18.998,'Ne':20.180,'Na':22.990,'Mg':24.305,

    'Al':26.982,'Si':28.086,'P':30.974,'S':32.065,'Cl':35.453,'Ar':39.948,

    'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,

    'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38

}

with open('model.xyz') as f:

    natoms = int(f.readline())

    f.readline()

    elems = [f.readline().split()[0] for _ in range(natoms)]

mass_g = sum(ATOMIC_MASS[e] for e in elems) * amu2g

box_df = pd.read_csv('thermo.out', sep=r'\s+', header=None, usecols=[9,13,17])

vol_cm3 = np.prod(box_df.values, axis=1) * ang3cm3

rho_gcm3 = mass_g / vol_cm3



                              

time = (np.arange(len(df)) + 1) * 10           



                                                      

COLORS = {

    "blue": "#609CC8",                          

    "deep_blue": "#3E79A8",

    "orange": "#F3A65A",                           

    "green": "#6AAE75",                              

    "red": "#C96B6B",                       

    "purple": "#8E79B9",          

    "brown": "#A78A7F",           

    "gray": "#BDBDBD",

    "dark_gray": "#4D4D4D",

}



plt.rcParams.update({

    "font.family": "Arial",

    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 12,

    "axes.labelsize": 13,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    "axes.linewidth": 1.1,

    "axes.grid": True,

    "grid.alpha": 0.25,

    "grid.linestyle": "--",

    "grid.linewidth": 0.7,

    "grid.color": "#BDBDBD",

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,

    "xtick.major.size": 4.5,

    "ytick.major.size": 4.5,

    "xtick.major.width": 1.0,

    "ytick.major.width": 1.0,

    "xtick.minor.size": 2.5,

    "ytick.minor.size": 2.5,

    "xtick.minor.width": 0.8,

    "ytick.minor.width": 0.8,

    "legend.frameon": False,

    "legend.fontsize": 10,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "savefig.dpi": 300,

})





def style_axis(ax):

    """Apply a consistent axis style."""

    ax.tick_params(axis="both", which="both", direction="in", width=1.0, length=4.5)

    for spine in ax.spines.values():

        spine.set_linewidth(1.1)

        spine.set_color(COLORS["dark_gray"])





def save_png_pdf(fig, filename_base):

    fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")

    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")





                               

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)



                  

ax1_t = ax1

ax1_k = ax1_t.twinx()

ax1_t.plot(time, df[0], color=COLORS['blue'], lw=1.35, alpha=0.88, label='Temperature')

ax1_k.plot(time, df[1], color=COLORS['orange'], lw=1.35, alpha=0.88, ls='--', label='Kinetic energy')

ax1_t.set_ylabel('Temperature (K)', color=COLORS['blue'])

ax1_k.set_ylabel('Kinetic energy (eV)', color=COLORS['orange'])

style_axis(ax1_t)

style_axis(ax1_k)

ax1_t.tick_params(axis='y', labelcolor=COLORS['blue'])

ax1_k.tick_params(axis='y', labelcolor=COLORS['orange'])

lines1, labels1 = ax1_t.get_legend_handles_labels()

lines2, labels2 = ax1_k.get_legend_handles_labels()

ax1_t.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, handlelength=1.6)



                  

ax2_u = ax2

ax2_rho = ax2_u.twinx()

ax2_u.plot(time, df[2], color=COLORS['green'], lw=1.35, alpha=0.88, label='Potential energy')

ax2_rho.plot(time, rho_gcm3, color=COLORS['red'], lw=1.35, alpha=0.88, label='Density')

ax2_u.set_ylabel('Potential energy (eV)', color=COLORS['green'])

ax2_rho.set_ylabel(r'Density (g cm$^{-3}$)', color=COLORS['red'])

style_axis(ax2_u)

style_axis(ax2_rho)

ax2_u.tick_params(axis='y', labelcolor=COLORS['green'])

ax2_rho.tick_params(axis='y', labelcolor=COLORS['red'])

lines1, labels1 = ax2_u.get_legend_handles_labels()

lines2, labels2 = ax2_rho.get_legend_handles_labels()

ax2_u.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=False, handlelength=1.6)



      

press_labels = ['Pxx', 'Pyy', 'Pzz', 'Pyz', 'Pxz', 'Pxy']

press_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'], COLORS['purple'], COLORS['brown']]

for icol, lab, col in zip(range(3, 9), press_labels, press_colors):

    ax3.plot(time, df[icol], lw=1.25, alpha=0.82, label=lab, color=col)

ax3.set_xlabel('Time (ps)')

ax3.set_ylabel('Pressure (GPa)')

style_axis(ax3)

ax3.legend(loc='lower right', ncol=2, frameon=False, handlelength=1.6, columnspacing=0.9)



plt.tight_layout()

save_png_pdf(fig, 'thermo_evolution')

plt.close(fig)          
