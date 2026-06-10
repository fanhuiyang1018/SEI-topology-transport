                     

                       

"""
Usage: python plot_li_msd.py <dt1_ps> <dt2_ps>
The two arguments define the time excluded from both ends of the linear MSD fit in ps.
"""

import sys

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress

import os

from pathlib import Path



                            

if len(sys.argv) != 3:

    sys.exit("Usage: python plot_MSD.py <dt1_ps> <dt2_ps>")

dt1, dt2 = float(sys.argv[1]), float(sys.argv[2])



                                        

elem_col = {

    'Li': {'msd': [19, 20, 21], 'sdc': [22, 23, 24], 'idx': 1},                    

}



                            

SCALE_FACTOR = 1                         



                                          

current_dir = Path.cwd()

msd_file = current_dir / "msd.out"



if not msd_file.exists():

    sys.exit(f"Error: msd.out was not found in the current directory ({current_dir.name}).")



print(f"Processing msd.out in the current directory.")



                                                   

COLORS = {

    "blue": "#609CC8",            

    "deep_blue": "#3E79A8",                  

    "orange": "#F3A65A",          

    "green": "#6AAE75",            

    "red": "#C96B6B",             

    "gray": "#BDBDBD",

    "dark_gray": "#4D4D4D",

}



plt.rcParams.update({

    "font.family": "Arial",

    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 11,

    "axes.labelsize": 12,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    "legend.fontsize": 10,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,

    "axes.linewidth": 1.1,

    "legend.frameon": False,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "savefig.dpi": 300,

})





def style_axis(ax):

    """Apply a consistent axis style."""

    ax.tick_params(which="both", direction="in", top=True, right=True, length=4.5, width=1.0)

    for spine in ax.spines.values():

        spine.set_linewidth(1.1)

        spine.set_color(COLORS["dark_gray"])





def save_png_pdf(fig, png_path):

    """Save a figure as both PNG and PDF."""

    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")





                           

try:

    data = np.loadtxt(msd_file)

except Exception as e:

    sys.exit(f"Error: failed to read {msd_file} Error message: {e}")



      

t_original = data[:, 0]



                                 

t_scaled = t_original * SCALE_FACTOR



        

elem = 'Li'

dic = elem_col[elem]

                

msdx_original, msdy_original, msdz_original = [data[:, i] for i in dic['msd']]

sdcx, sdcy, sdcz = [data[:, i] for i in dic['sdc']]



                     

msdx_scaled = msdx_original * SCALE_FACTOR

msdy_scaled = msdy_original * SCALE_FACTOR

msdz_scaled = msdz_original * SCALE_FACTOR

msd_scaled = msdx_scaled + msdy_scaled + msdz_scaled



                                     

msdx_scaled_zeroed = msdx_scaled - msdx_scaled[0]

msdy_scaled_zeroed = msdy_scaled - msdy_scaled[0]

msdz_scaled_zeroed = msdz_scaled - msdz_scaled[0]

msd_scaled_zeroed = msd_scaled - msd_scaled[0]



sdc_avg = (sdcx + sdcy + sdcz) / 3.0



                               

sdcx_SI = sdcx * 1e-4

sdcy_SI = sdcy * 1e-4

sdcz_SI = sdcz * 1e-4

sdc_avg_SI = sdc_avg * 1e-4



                       

t1_scaled, t2_scaled = t_scaled.min() + dt1 * SCALE_FACTOR, t_scaled.max() - dt2 * SCALE_FACTOR

mask = (t_scaled >= t1_scaled) & (t_scaled <= t2_scaled)

if t_scaled[mask].size == 0:

    sys.exit(f"Error: no data remain after applying the fit window.")



                                

slope, intercept, r_val, *_ = linregress(t_scaled[mask], msd_scaled_zeroed[mask])

D_msd = slope / 6.0 * 1e-4          



                                    

summary_file = current_dir / f'diffusion_fit_summary_{elem}.txt'

with open(summary_file, 'w') as f:

    f.write(f'{current_dir.name} - {elem} diffusion coefficient (MSD linear fit)\n')

    f.write(f'Fit range: {t1_scaled/SCALE_FACTOR:.1f}–{t2_scaled/SCALE_FACTOR:.1f} ps (original), ')

    f.write(f'{t1_scaled:.1f}–{t2_scaled:.1f} ps (scaled x{SCALE_FACTOR})\n')

    f.write(f'Scaling: time and MSD both multiplied by {SCALE_FACTOR}, slope unchanged\n')

    f.write(f'# MSD zeroed: initial values subtracted (MSD(t) - MSD(0))\n')

    f.write(f'D = {D_msd:.3e} cm^2/s\n')

    f.write(f'slope = {slope:.3e} A^2/ps (scaled, zeroed)\n')

    f.write(f'original slope = {slope/SCALE_FACTOR:.3e} A^2/ps\n')

    f.write(f'intercept = {intercept:.3e} A^2 (zeroed)\n')

    f.write(f'R^2 = {r_val**2:.4f}\n\n')

    f.write(f'# Self-diffusion coefficient (SDC average)\n')

    f.write(f'SDC_x = {sdcx_SI[mask].mean():.3e} cm^2/s\n')

    f.write(f'SDC_y = {sdcy_SI[mask].mean():.3e} cm^2/s\n')

    f.write(f'SDC_z = {sdcz_SI[mask].mean():.3e} cm^2/s\n')

    f.write(f'SDC_avg = {sdc_avg_SI[mask].mean():.3e} cm^2/s\n')



                                

fig1, ax1 = plt.subplots(figsize=(6, 4.5))



ax1.plot(t_scaled[mask], msdx_scaled_zeroed[mask], lw=1.35, color=COLORS["blue"],

         alpha=0.82, label='MSD$_x$')

ax1.plot(t_scaled[mask], msdy_scaled_zeroed[mask], lw=1.35, color=COLORS["orange"],

         alpha=0.82, label='MSD$_y$')

ax1.plot(t_scaled[mask], msdz_scaled_zeroed[mask], lw=1.35, color=COLORS["green"],

         alpha=0.82, label='MSD$_z$')

ax1.plot(t_scaled[mask], msd_scaled_zeroed[mask], lw=1.8, color=COLORS["deep_blue"],

         alpha=0.90, label='MSD$_\\mathrm{total}$')

msd_line = slope * t_scaled[mask] + intercept

ax1.plot(t_scaled[mask], msd_line, '--', color=COLORS["red"], lw=1.30,

         alpha=0.88, label=f'fit: D={D_msd:.2e} cm$^2$/s')



ax1.set_xlim(t1_scaled, t2_scaled)

ax1.set_ylim(bottom=0)

ax1.set_xlabel(f'Time / ps')

ax1.set_ylabel(f'MSD / Å²')

                                          

ax1.legend(loc='upper left', frameon=False)



plt.tight_layout()

msd_fig_file = current_dir / f'msd_{elem}.png'

plt.savefig(msd_fig_file, dpi=300)

plt.savefig(msd_fig_file.with_suffix('.pdf'), bbox_inches='tight')

plt.close(fig1)



                                

fig2, ax2 = plt.subplots(figsize=(6, 4.5))



ax2.plot(t_scaled, sdcx_SI, lw=1.35, color=COLORS["blue"],

         alpha=0.82, label='SDC$_x$')

ax2.plot(t_scaled, sdcy_SI, lw=1.35, color=COLORS["orange"],

         alpha=0.82, label='SDC$_y$')

ax2.plot(t_scaled, sdcz_SI, lw=1.35, color=COLORS["green"],

         alpha=0.82, label='SDC$_z$')

ax2.plot(t_scaled, sdc_avg_SI, lw=1.8, color=COLORS["deep_blue"],

         alpha=0.90, label='SDC$_\\mathrm{avg}$')



ax2.set_xlim(left=0)

ax2.set_xlabel(f'Time / ps')

ax2.set_ylabel('SDC / cm² s⁻¹')

                                          

ax2.legend(loc='upper left', frameon=False)



plt.tight_layout()

sdc_fig_file = current_dir / f'sdc_{elem}.png'

plt.savefig(sdc_fig_file, dpi=300)

plt.savefig(sdc_fig_file.with_suffix('.pdf'), bbox_inches='tight')

plt.close(fig2)



print(f'\nProcessing complete. Generated files in the current directory:')

print(f'  - msd_{elem}.png / msd_{elem}.pdf (zero-shifted MSD)')

print(f'  - sdc_{elem}.png / sdc_{elem}.pdf (self-diffusion coefficient)')

print(f'  - diffusion_fit_summary_{elem}.txt (fit summary)')

print(f'Note: time and MSD are scaled by {SCALE_FACTOR}; MSD is shifted to zero at t=0, and the diffusion coefficient remains unchanged.')
