import numpy as np

import re

import matplotlib.pyplot as plt

from matplotlib import rcParams



                                                  

XYZ_PATH = "positions.xyz"

LI_SYMBOL = "Li"

F_SYMBOL = "F"

SKIP_STEPS = 0



                              

TOTAL_TIME_PS = 10000.0



        

BOND_MAX = 2.8

ANGLE_TOL_TETRA = 25.0

ANGLE_TOL_OCTA = 30.0



       

TETRA_ANGLE_IDEAL = 109.5

TETRA_ANGLE_MIN = TETRA_ANGLE_IDEAL - ANGLE_TOL_TETRA

TETRA_ANGLE_MAX = TETRA_ANGLE_IDEAL + ANGLE_TOL_TETRA



       

OCTA_ANGLE_90_IDEAL = 90.0

OCTA_ANGLE_180_IDEAL = 180.0

OCTA_ANGLE_90_MIN = OCTA_ANGLE_90_IDEAL - ANGLE_TOL_OCTA

OCTA_ANGLE_90_MAX = OCTA_ANGLE_90_IDEAL + ANGLE_TOL_OCTA

OCTA_ANGLE_180_MIN = OCTA_ANGLE_180_IDEAL - ANGLE_TOL_OCTA

OCTA_ANGLE_180_MAX = OCTA_ANGLE_180_IDEAL + ANGLE_TOL_OCTA



              

OUTPUT_CONV_FREQ = "motif_conversion_frequency.png"

OUTPUT_SURVIVAL = "motif_survival_probability.png"

OUTPUT_SURVIVAL_ZOOM = "motif_survival_probability_0_10ps.png"

OUTPUT_POPULATION = "motif_population_timeseries.png"



                                         

SURVIVAL_ZOOM_TIME_PS = 100



                                                        

COLORS = {

    "tetra": "#F3A65A",                   

    "octa": "#609CC8",                   

    "other": "#CFCFCF",             

    "dark_gray": "#4D4D4D",

}



rcParams.update({

    "font.family": "Arial",

    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "axes.unicode_minus": False,

    "font.size": 14,

    "axes.labelsize": 24,

    "xtick.labelsize": 20,

    "ytick.labelsize": 20,

    "legend.fontsize": 14,

    "axes.linewidth": 1.1,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": False,

    "ytick.right": False,

    "xtick.major.width": 1.0,

    "ytick.major.width": 1.0,

    "xtick.major.size": 4.5,

    "ytick.major.size": 4.5,

    "legend.frameon": False,

    "savefig.dpi": 300,

})



STATE_OTHER = 0

STATE_TETRA = 1

STATE_OCTA = 2



STATE_NAME = {

    STATE_OTHER: "Other",

    STATE_TETRA: "Tetrahedron",

    STATE_OCTA: "Octahedron",

}





def style_axis(ax):

    ax.tick_params(axis="both", which="major", direction="in", width=1.0, length=4.5)

    for spine in ax.spines.values():

        spine.set_linewidth(1.1)

        spine.set_color(COLORS["dark_gray"])





def is_tetrahedron_loose(vecs):

    """Return True if the neighbor angles are compatible with a tetrahedral motif."""

    if len(vecs) != 4:

        return False



    angles = []

    for i in range(4):

        for j in range(i + 1, 4):

            v1, v2 = vecs[i], vecs[j]

            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if norm1 < 1e-8 or norm2 < 1e-8:

                return False

            cos_ang = np.dot(v1, v2) / (norm1 * norm2)

            cos_ang = np.clip(cos_ang, -1.0, 1.0)

            ang = np.degrees(np.arccos(cos_ang))

            angles.append(ang)



    return all(TETRA_ANGLE_MIN <= a <= TETRA_ANGLE_MAX for a in angles)





def is_octahedron_loose(vecs):

    """Return True if the neighbor angles are compatible with an octahedral motif."""

    if len(vecs) != 6:

        return False



    angles = []

    for i in range(6):

        for j in range(i + 1, 6):

            v1, v2 = vecs[i], vecs[j]

            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if norm1 < 1e-8 or norm2 < 1e-8:

                continue

            cos_ang = np.dot(v1, v2) / (norm1 * norm2)

            cos_ang = np.clip(cos_ang, -1.0, 1.0)

            ang = np.degrees(np.arccos(cos_ang))

            angles.append(ang)



    if len(angles) == 0:

        return False



    near_90 = sum(1 for ang in angles if OCTA_ANGLE_90_MIN <= ang <= OCTA_ANGLE_90_MAX)

    near_180 = sum(1 for ang in angles if OCTA_ANGLE_180_MIN <= ang <= OCTA_ANGLE_180_MAX)



    total_angles = len(angles)

    fraction_90 = near_90 / total_angles if total_angles > 0 else 0.0



    return (fraction_90 > 0.4) and (near_180 >= 2)





def classify_one_li(li_xyz, f_pos):

    """Classify the local environment of one Li ion."""

    distances = np.linalg.norm(f_pos - li_xyz, axis=1)

    f_near_indices = np.where(distances < BOND_MAX)[0]

    n_coord = len(f_near_indices)



    if n_coord == 4:

        vecs = f_pos[f_near_indices] - li_xyz

        if is_tetrahedron_loose(vecs):

            return STATE_TETRA

        return STATE_OTHER



    elif n_coord == 6:

        vecs = f_pos[f_near_indices] - li_xyz

        if is_octahedron_loose(vecs):

            return STATE_OCTA

        return STATE_OTHER



    else:

        return STATE_OTHER





def parse_xyz_frames():

    """Read an XYZ trajectory and return Li and F coordinates for each frame."""

    with open(XYZ_PATH, "r", encoding="utf-8") as f:

        lines = [l.strip() for l in f if l.strip()]



    total_atoms = int(lines[0])

    frame_lines = 1 + 1 + total_atoms

    total_frames = len(lines) // frame_lines



    print(f"Total atoms: {total_atoms}")

    print(f"Total frames: {total_frames}")

    print(f"Skipped first {SKIP_STEPS} frames")



    li_frames = []

    f_frames = []



    for frame in range(SKIP_STEPS, total_frames):

        idx0 = frame * frame_lines

        atom_lines = lines[idx0 + 2: idx0 + 2 + total_atoms]



        li_pos = []

        f_pos = []



        for line in atom_lines:

            s = line.split()

            sym = s[0]

            xyz = np.array([float(s[1]), float(s[2]), float(s[3])], dtype=np.float64)



            if sym == LI_SYMBOL:

                li_pos.append(xyz)

            elif sym == F_SYMBOL:

                f_pos.append(xyz)



        li_frames.append(np.array(li_pos))

        f_frames.append(np.array(f_pos))



    return li_frames, f_frames





def build_state_matrix(li_frames, f_frames):

    """Build the motif-state matrix with shape [frame, Li index]."""

    n_frames = len(li_frames)

    n_li = len(li_frames[0])



    state_matrix = np.zeros((n_frames, n_li), dtype=np.int8)



    for frame in range(n_frames):

        li_pos = li_frames[frame]

        f_pos = f_frames[frame]



        for i, li_xyz in enumerate(li_pos):

            state_matrix[frame, i] = classify_one_li(li_xyz, f_pos)



        if frame % max(1, n_frames // 10) == 0:

            tetra_count = np.sum(state_matrix[frame] == STATE_TETRA)

            octa_count = np.sum(state_matrix[frame] == STATE_OCTA)

            other_count = np.sum(state_matrix[frame] == STATE_OTHER)

            print(f"Frame {frame:4d} | Tetra: {tetra_count:4d} | Octa: {octa_count:4d} | Other: {other_count:4d}")



    return state_matrix





def build_time_axis(n_frames):

    """Build the trajectory time axis from the total simulation time."""

    if n_frames == 1:

        return np.array([0.0])

    return np.linspace(0.0, TOTAL_TIME_PS, n_frames)





def calculate_population_timeseries(state_matrix):

    tetra_counts = np.sum(state_matrix == STATE_TETRA, axis=1)

    octa_counts = np.sum(state_matrix == STATE_OCTA, axis=1)

    other_counts = np.sum(state_matrix == STATE_OTHER, axis=1)

    return tetra_counts, octa_counts, other_counts





def extract_residence_events(state_matrix, time_ps):

    """
    Extract residence events for each Li ion and return them as a list of dictionaries.
    """

    n_frames, n_li = state_matrix.shape

    dt_ps = time_ps[1] - time_ps[0] if n_frames > 1 else TOTAL_TIME_PS



    events = []



    for li_idx in range(n_li):

        states = state_matrix[:, li_idx]



        start = 0

        current_state = states[0]



        for f in range(1, n_frames):

            if states[f] != current_state:

                end = f - 1

                duration_frames = end - start + 1

                duration_ps = duration_frames * dt_ps



                prev_state = states[start - 1] if start > 0 else -1

                next_state = states[f]



                events.append({

                    "li_index": li_idx,

                    "state": current_state,

                    "start_frame": start,

                    "end_frame": end,

                    "start_time_ps": time_ps[start],

                    "end_time_ps": time_ps[end],

                    "duration_ps": duration_ps,

                    "prev_state": prev_state,

                    "next_state": next_state,

                })



                start = f

                current_state = states[f]



              

        end = n_frames - 1

        duration_frames = end - start + 1

        duration_ps = duration_frames * dt_ps

        prev_state = states[start - 1] if start > 0 else -1

        next_state = -1



        events.append({

            "li_index": li_idx,

            "state": current_state,

            "start_frame": start,

            "end_frame": end,

            "start_time_ps": time_ps[start],

            "end_time_ps": time_ps[end],

            "duration_ps": duration_ps,

            "prev_state": prev_state,

            "next_state": next_state,

        })



    return events





def get_state_durations(events, target_state):

    return np.array([ev["duration_ps"] for ev in events if ev["state"] == target_state], dtype=np.float64)





def compute_survival_probability(durations, max_time_ps, n_points=500):

    """
    Survival probability S(t) = P(duration >= t)
    Plot the complete survival curve without truncation.
    """

    thresholds = np.linspace(0.0, max_time_ps, n_points)



    if len(durations) == 0:

        survival = np.zeros_like(thresholds)

        return thresholds, survival



    survival = np.array([np.mean(durations >= t) for t in thresholds], dtype=np.float64)

    return thresholds, survival





def compute_conversion_frequencies(events, n_li, total_time_ps):

    """
    Count T-to-O and O-to-T conversion frequencies in Li^-1 ns^-1.
    """

    t_to_o = 0

    o_to_t = 0



    for ev in events:

        if ev["state"] == STATE_TETRA and ev["next_state"] == STATE_OCTA:

            t_to_o += 1

        elif ev["state"] == STATE_OCTA and ev["next_state"] == STATE_TETRA:

            o_to_t += 1



    total_time_ns = total_time_ps / 1000.0



    freq_t_to_o = t_to_o / (n_li * total_time_ns) if total_time_ns > 0 else 0.0

    freq_o_to_t = o_to_t / (n_li * total_time_ns) if total_time_ns > 0 else 0.0



    return freq_t_to_o, freq_o_to_t





def plot_survival_probability(t_tetra, s_tetra, t_octa, s_octa, output_path=OUTPUT_SURVIVAL, x_max_ps=TOTAL_TIME_PS):

    fig, ax = plt.subplots(figsize=(6.2, 4.4))



    ax.plot(

        t_tetra, s_tetra,

        color=COLORS["tetra"],

        linewidth=1.5,

        alpha=0.92,

        label="Tetrahedron"

    )

    ax.plot(

        t_octa, s_octa,

        color=COLORS["octa"],

        linewidth=1.5,

        alpha=0.92,

        label="Octahedron"

    )



    ax.set_xlabel("Residence time threshold (ps)")

    ax.set_ylabel("Survival probability")

    ax.set_xlim(0, x_max_ps)

    ax.set_ylim(-0.02, 1.02)



    style_axis(ax)

    ax.legend(loc="best", frameon=False)

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_path}")





def plot_conversion_frequency(freq_t_to_o, freq_o_to_t):

    fig, ax = plt.subplots(figsize=(4.2, 4.2))



    labels = ["T to O", "O to T"]

    values = [freq_t_to_o, freq_o_to_t]

    colors = [COLORS["tetra"], COLORS["octa"]]



    ax.bar(labels, values, color=colors, width=0.65, alpha=0.95)



    ax.set_ylabel(r"Frequency ($\mathrm{Li^{-1}\ ns^{-1}}$)")

    style_axis(ax)



    plt.tight_layout()

    fig.savefig(OUTPUT_CONV_FREQ, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {OUTPUT_CONV_FREQ}")





def plot_population_timeseries(time_ps, tetra_counts, octa_counts, other_counts):

    """
    Plot motif populations with a broken y axis.
    """

    fig, (ax_top, ax_bottom) = plt.subplots(

        2, 1, sharex=True, figsize=(6.5, 4.8),

        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05}

    )



    for ax in [ax_top, ax_bottom]:

        ax.plot(time_ps, tetra_counts, color=COLORS["tetra"], linewidth=1.3, alpha=0.9, label="Tetrahedron")

        ax.plot(time_ps, octa_counts, color=COLORS["octa"], linewidth=1.3, alpha=0.9, label="Octahedron")

        ax.plot(time_ps, other_counts, color=COLORS["other"], linewidth=1.3, alpha=0.95, label="Other")

        style_axis(ax)

        ax.tick_params(axis="x", which="both", top=False, bottom=True)

        ax.tick_params(axis="y", which="both", right=False, left=True)



                     

    bottom_ylim = (-50, 2000)

    top_ylim = (3300, max(np.max(octa_counts), np.max(other_counts), np.max(tetra_counts)) + 100)

    ax_bottom.set_ylim(*bottom_ylim)

    ax_top.set_ylim(*top_ylim)



                

    ax_top.spines["bottom"].set_visible(False)

    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_bottom.xaxis.tick_bottom()



                                          

    bottom_ticks = [tick for tick in ax_bottom.get_yticks() if bottom_ylim[0] <= tick < bottom_ylim[1]]

    top_ticks = [tick for tick in ax_top.get_yticks() if top_ylim[0] < tick <= top_ylim[1]]

    ax_bottom.set_yticks(bottom_ticks)

    ax_top.set_yticks(top_ticks)



            

    d = 0.012

    kwargs = dict(transform=ax_top.transAxes, color=COLORS["dark_gray"], clip_on=False, linewidth=1.0)

    ax_top.plot((-d, +d), (-d, +d), **kwargs)

    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)



    kwargs.update(transform=ax_bottom.transAxes)

    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)



    ax_bottom.set_xlabel("Time (ps)")

    ax_bottom.set_xlim(0, TOTAL_TIME_PS)



                    

    ax_top.set_ylabel("")

    ax_bottom.set_ylabel("")

    fig.supylabel("Number per cell", fontsize=24, x=0.02)



    ax_top.legend(loc="center right", frameon=False)



    fig.tight_layout()

    fig.subplots_adjust(left=0.16)

    fig.savefig(OUTPUT_POPULATION, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {OUTPUT_POPULATION}")





                                                 

if __name__ == "__main__":

    print("=" * 70)

    print("Li motif lifetime / conversion analysis")

    print(f"XYZ file: {XYZ_PATH}")

    print(f"Total physical time: {TOTAL_TIME_PS} ps (10 ns)")

    print(f"Bond cutoff: {BOND_MAX} Å")

    print(f"Tetrahedral angular tolerance: ±{ANGLE_TOL_TETRA}°")

    print(f"Octahedral angular tolerance: ±{ANGLE_TOL_OCTA}°")

    print("=" * 70)



             

    li_frames, f_frames = parse_xyz_frames()



               

    state_matrix = build_state_matrix(li_frames, f_frames)

    n_frames, n_li = state_matrix.shape

    time_ps = build_time_axis(n_frames)



                              

    tetra_counts, octa_counts, other_counts = calculate_population_timeseries(state_matrix)



                         

    events = extract_residence_events(state_matrix, time_ps)



                             

    tetra_durations = get_state_durations(events, STATE_TETRA)

    octa_durations = get_state_durations(events, STATE_OCTA)



    t_tetra, s_tetra = compute_survival_probability(tetra_durations, TOTAL_TIME_PS, n_points=1000)

    t_octa, s_octa = compute_survival_probability(octa_durations, TOTAL_TIME_PS, n_points=1000)



                                             

                                                    

                                      

    t_tetra_zoom, s_tetra_zoom = compute_survival_probability(

        tetra_durations, SURVIVAL_ZOOM_TIME_PS, n_points=500

    )

    t_octa_zoom, s_octa_zoom = compute_survival_probability(

        octa_durations, SURVIVAL_ZOOM_TIME_PS, n_points=500

    )



                             

    freq_t_to_o, freq_o_to_t = compute_conversion_frequencies(events, n_li, TOTAL_TIME_PS)



             

    plot_survival_probability(t_tetra, s_tetra, t_octa, s_octa)

    plot_survival_probability(

        t_tetra_zoom, s_tetra_zoom,

        t_octa_zoom, s_octa_zoom,

        output_path=OUTPUT_SURVIVAL_ZOOM,

        x_max_ps=SURVIVAL_ZOOM_TIME_PS

    )

    plot_conversion_frequency(freq_t_to_o, freq_o_to_t)

    plot_population_timeseries(time_ps, tetra_counts, octa_counts, other_counts)



               

    print("\n" + "=" * 70)

    print("Summary:")

    print(f"Number of Li ions: {n_li}")

    print(f"Number of frames: {n_frames}")

    print(f"Average tetrahedral population: {np.mean(tetra_counts):.2f}")

    print(f"Average octahedral population: {np.mean(octa_counts):.2f}")

    print(f"Average other population: {np.mean(other_counts):.2f}")

    print(f"T-to-O frequency: {freq_t_to_o:.6e} Li^-1 ns^-1")

    print(f"O-to-T frequency: {freq_o_to_t:.6e} Li^-1 ns^-1")

    print("=" * 70)

    print("Analysis complete.")
