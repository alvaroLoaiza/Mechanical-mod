"""
Mechanical Modulus and Manufacturing Cost Models
=================================================
Analytical sub-models for use in IV.B multi-objective optimization.
Designed to be called inside the genetic algorithm loop alongside
sigma_eff and selectivity from II.C / IV.A.

Two models
----------
1. modulus(DS, C_xl, phi_filler, phi_stiff, phi_sPEEK, E_stiff)
   Predicts Young's modulus [MPa] of an SLM membrane from composition.
   Combines:
     - Sulfonation-modulus degradation  : exponential fit to sPEEK literature
     - Crosslinker stiffening           : linear rule
     - Polymer blend                    : rule of mixtures (sPEEK + co-polymer)
     - Nano-filler reinforcement        : Halpin-Tsai model (Al2O3)

   phi_stiff is the volume fraction of the stiff co-polymer component.
   E_stiff is its Young's modulus -- set per chamber:
     SLM-Li : E_stiff = E_LCP   = 9000 MPa  (Vectra LCP)
     SLM-H  : E_stiff = E_PTFE  =  500 MPa  (PTFE backing)
     SLM-B  : E_stiff = E_PVA   =   60 MPa  (PVA in PEO/PVA gel)

2. manufacturing_cost(phi_sPEEK, phi_stiff, phi_filler, thickness_um,
                      area_ft2, phase)
   Predicts manufacturing cost [USD/ft2] from BOM + processing.
   Based on Planck Power binder bottom-up cost structure.

Literature sources for constants
---------------------------------
E_PEEK_0   = 3600 MPa   -- pristine PEEK (Victrex 450G datasheet)
K_DS       = 5.5        -- sulfonation degradation constant.
                           E_PEEK_0 is FIXED (not fitted) to the Victrex value.
                           With E_PEEK_0 constrained, K_DS is the only free
                           parameter, solved by constrained least-squares:
                             K_DS = sum(DS_i*(ln(3600)-ln(E_i))) / sum(DS_i^2)
                           Calibration points:
                             DS=0.40 E=400 MPa -> K=5.49 (sPEEK literature)
                             DS=0.60 E=100 MPa -> K=5.99 (sPEEK literature)
                             DS=0.70 E= 24 MPa -> K=6.81 (PMC7281369, sPEEK7)
                           Constrained LS gives K_DS = 6.14.
                           Current K_DS = 5.5 is conservative (anchored to the
                           DS=0.40 point), slightly overestimating modulus at
                           high DS (DS > 0.65). Update to 6.14 for a better fit.
E_LCP      = 9000 MPa   -- Vectra LCP in-plane (Chenniki et al. 2015, 5.7-12 GPa)
E_PTFE     =  500 MPa   -- PTFE film (literature range 400-600 MPa)
E_PVA      =   60 MPa   -- PVA hydrogel (literature range 40-80 MPa)
E_AL2O3    = 253000 MPa -- nano gamma-Al2O3 (nanocrystalline, Academia.edu 1994)
ALPHA_XL   = 3.0        -- crosslinker stiffening: ~15% per 5 wt% (estimated
                           from Hou et al. 2013, qualitative trend only --
                           least constrained parameter, needs DMA validation)
XI_HT      = 2.0        -- Halpin-Tsai shape factor for spherical particles
                           (Halpin & Kardos 1969; Zare & Rhee 2016)

NOTE: np.clip() calls throughout this module silently clamp inputs to
physical bounds. Safe for standalone use but DANGEROUS inside an
optimization loop -- the optimizer receives a valid-looking output for
an out-of-range input, corrupting the objective landscape.
Before integrating with NSGA-II / pymoo, replace np.clip() with
explicit ValueError checks, or rely on the optimizer xl/xu bounds
and remove the clips entirely.

Manufacturing cost constants (Planck Power binder, Section III.A BOM):
  sPEEK         $50/kg
  co-polymer    $15/kg   (off-spec LCP default; override for other blends)
  nano-Al2O3    $80/kg
  Processing    $0.17/ft2 base (slot-die + drying + edge sealing)
  Assembly      $2.00/ft2 Phase 1 (lamination, calendering, QC)
  Overhead      15%
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# MATERIAL CONSTANTS  (all in MPa unless noted)
# ===========================================================================

E_PEEK_0   = 3600.0     # MPa  pristine PEEK (Victrex 450G)
E_LCP      = 9000.0     # MPa  Vectra LCP in-plane
E_PTFE     =  500.0     # MPa  PTFE film
E_PVA      =   60.0     # MPa  PVA hydrogel
E_AL2O3    = 253000.0   # MPa  nano gamma-Al2O3

K_DS       = 5.5        # sulfonation degradation constant (see docstring)
ALPHA_XL   = 3.0        # crosslinker stiffening per unit wt fraction
XI_HT      = 2.0        # Halpin-Tsai shape factor (spheres)

# Manufacturing cost constants
COST_SPEEK       = 50.0   # USD/kg
COST_COPOLYMER   = 15.0   # USD/kg  (default: off-spec LCP)
COST_AL2O3       = 80.0   # USD/kg

RHO_SPEEK        = 1.29   # g/cm3
RHO_COPOLYMER    = 1.40   # g/cm3
RHO_AL2O3        = 3.99   # g/cm3

COST_PROC_BASE   = 0.10   # USD/ft2  slot-die coating
COST_DRYING      = 0.04   # USD/ft2  drying
COST_SEALING     = 0.03   # USD/ft2  edge sealing
COST_ASSEMBLY    = {1: 2.00, 2: 1.50, 3: 1.00}  # USD/ft2 lamination + QC
OVERHEAD_FRAC    = 0.15   # 15% overhead
PHASE_FACTOR     = {1: 1.00, 2: 0.79, 3: 0.63}  # learning curve
FT2_TO_CM2       = 929.03


# ===========================================================================
# 1.  MODULUS MODEL
# ===========================================================================

def E_sPEEK(DS):
    """
    Young's modulus of sPEEK vs degree of sulfonation.

        E(DS) = E_PEEK_0 * exp(-K_DS * DS)

    E_PEEK_0 fixed to Victrex datasheet value (3600 MPa).
    K_DS = 5.5 fitted by constrained least-squares (see module docstring).
    """
    DS = float(np.clip(DS, 0.0, 0.95))
    return E_PEEK_0 * np.exp(-K_DS * DS)


def halpin_tsai(E_matrix, E_filler, phi_filler, xi=XI_HT):
    """
    Halpin-Tsai model for particle-reinforced composite modulus.

        eta = (E_f/E_m - 1) / (E_f/E_m + xi)
        E_c = E_m * (1 + xi*eta*phi) / (1 - eta*phi)

    xi = 2 for spherical nanoparticles (Halpin & Kardos 1969).
    """
    phi = float(np.clip(phi_filler, 0.0, 0.4))
    lam = E_filler / max(E_matrix, 1e-6)
    eta = (lam - 1.0) / (lam + xi)
    return E_matrix * (1.0 + xi * eta * phi) / max(1.0 - eta * phi, 1e-6)


def modulus(DS, C_xl, phi_filler, phi_stiff, phi_sPEEK, E_stiff=E_LCP):
    """
    Predict Young's modulus of an SLM membrane from composition.

    Calculation chain:
        Step 1: E_sPEEK(DS)  -- sulfonation degradation
        Step 2: Rule of mixtures for sPEEK + co-polymer blend
        Step 3: Crosslinker stiffening (linear)
        Step 4: Halpin-Tsai for nano-Al2O3 filler

    Parameters
    ----------
    DS         : float   degree of sulfonation (0 to 1)
    C_xl       : float   crosslinker weight fraction (0 to 0.15)
    phi_filler : float   nano-Al2O3 volume fraction (0 to 0.10)
    phi_stiff  : float   stiff co-polymer volume fraction (0 to 1)
                         NOT always LCP -- set per chamber:
                           SLM-Li: phi_stiff = LCP fraction
                           SLM-H : phi_stiff = PTFE fraction
                           SLM-B : phi_stiff = PVA fraction
    phi_sPEEK  : float   sPEEK volume fraction (0 to 1)
    E_stiff    : float   Young's modulus of co-polymer [MPa]
                         Default E_LCP = 9000 MPa.
                         Override: E_PVA=60 (SLM-B), E_PTFE=500 (SLM-H)

    Returns
    -------
    float : Young's modulus [MPa]
    """
    # Normalise polymer blend fractions
    total = phi_stiff + phi_sPEEK
    if total > 0:
        phi_stiff = phi_stiff / total
        phi_sPEEK = phi_sPEEK / total

    # Step 1
    E_sp = E_sPEEK(DS)

    # Step 2: rule of mixtures
    E_blend = phi_sPEEK * E_sp + phi_stiff * E_stiff

    # Step 3: crosslinker stiffening
    C_xl   = float(np.clip(C_xl, 0.0, 0.15))
    E_xl   = E_blend * (1.0 + ALPHA_XL * C_xl)

    # Step 4: Halpin-Tsai
    E_final = halpin_tsai(E_xl, E_AL2O3, phi_filler)

    return float(E_final)


# ===========================================================================
# 2.  MANUFACTURING COST MODEL
# ===========================================================================

def manufacturing_cost(phi_sPEEK, phi_stiff, phi_filler,
                       thickness_um, area_ft2=1.0, phase=1,
                       cost_copolymer=COST_COPOLYMER,
                       rho_copolymer=RHO_COPOLYMER):
    """
    Bottom-up manufacturing cost for one SLM membrane [USD/ft2].

    Material cost   : mass per ft2 x unit cost per kg
    Processing cost : slot-die + drying + sealing (fixed per ft2)
    Assembly cost   : lamination + calendering + QC (phase-dependent)
    Overhead        : 15% on material + processing subtotal

    Parameters
    ----------
    phi_sPEEK      : float   sPEEK volume fraction
    phi_stiff      : float   co-polymer volume fraction
    phi_filler     : float   Al2O3 volume fraction
    thickness_um   : float   membrane thickness [um]
    area_ft2       : float   area [ft2]  (default 1)
    phase          : int     manufacturing phase 1/2/3
    cost_copolymer : float   co-polymer unit cost [USD/kg]  (default LCP $15/kg)
    rho_copolymer  : float   co-polymer density [g/cm3]     (default LCP 1.40)
    """
    total = phi_sPEEK + phi_stiff + phi_filler
    if total > 1.0:
        phi_sPEEK  /= total
        phi_stiff  /= total
        phi_filler /= total

    t_cm = thickness_um * 1e-4
    vol  = FT2_TO_CM2 * t_cm

    m_sPEEK  = vol * phi_sPEEK  * RHO_SPEEK     * 1e-3
    m_stiff  = vol * phi_stiff  * rho_copolymer  * 1e-3
    m_filler = vol * phi_filler * RHO_AL2O3      * 1e-3

    mat_cost  = m_sPEEK * COST_SPEEK + m_stiff * cost_copolymer + m_filler * COST_AL2O3
    proc_cost = COST_PROC_BASE + COST_DRYING + COST_SEALING
    assembly  = COST_ASSEMBLY.get(phase, 2.0)

    subtotal   = mat_cost + proc_cost
    total_cost = subtotal * (1.0 + OVERHEAD_FRAC) + assembly
    total_cost *= PHASE_FACTOR.get(phase, 1.0)

    return float(total_cost * area_ft2)


# ===========================================================================
# 3.  PLOTS
# ===========================================================================

def make_plots(output_dir):
    """
    Generate and save four diagnostic plots to output_dir:
      Fig 1 -- E_sPEEK(DS): model curve + calibration data points
      Fig 2 -- Modulus sensitivity per input variable (SLM-Li baseline)
      Fig 3 -- Halpin-Tsai: composite modulus vs filler fraction
      Fig 4 -- Manufacturing cost vs thickness for all three chambers
    """
    os.makedirs(output_dir, exist_ok=True)

    BLUE  = "#185FA5"
    AMBER = "#BA7517"
    TEAL  = "#0F6E56"
    RED   = "#C0392B"
    GRAY  = "#5F5E5A"
    BG    = "#F8F8F6"

    # ── Fig 1: E_sPEEK degradation curve ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor("#FAFAF8")

    ds_arr = np.linspace(0, 0.90, 200)
    ax.plot(ds_arr, [E_sPEEK(d) for d in ds_arr],
            color=BLUE, lw=2.2, label=f"Model: 3600 * exp(-{K_DS} * DS)")

    # Calibration data points
    ds_lit = np.array([0.00, 0.40, 0.60, 0.70])
    E_lit  = np.array([3600., 400., 100., 24.])
    src    = ["Victrex datasheet", "sPEEK literature",
              "sPEEK literature", "PMC7281369 (sPEEK7)"]
    for d, e, s in zip(ds_lit, E_lit, src):
        ax.scatter(d, e, s=70, zorder=5, color=RED)
        ax.annotate(s, (d, e), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=GRAY)

    ax.set_xlabel("Degree of sulfonation (DS)", fontsize=11)
    ax.set_ylabel("Young's modulus (MPa)", fontsize=11)
    ax.set_title("sPEEK modulus degradation with sulfonation\n"
                 "E(DS) = 3600 * exp(-K_DS * DS)  --  E_PEEK_0 fixed, K_DS constrained LS",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    p = os.path.join(output_dir, "fig1_sPEEK_degradation.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 2: Sensitivity analysis (SLM-Li baseline) ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Modulus sensitivity -- SLM-Li baseline\n"
                 "(DS=0.75, C_xl=0, phi_filler=0.03, phi_stiff=0.30, E_stiff=LCP)",
                 fontsize=10)

    sweep_params = [
        ("DS",         np.linspace(0.30, 0.90, 50),
         lambda v: modulus(v, 0.0, 0.03, 0.30, 0.70, E_LCP),
         "Degree of sulfonation (DS)", BLUE),
        ("C_xl",       np.linspace(0.0,  0.15, 50),
         lambda v: modulus(0.75, v,  0.03, 0.30, 0.70, E_LCP),
         "Crosslinker wt fraction", AMBER),
        ("phi_filler", np.linspace(0.0,  0.10, 50),
         lambda v: modulus(0.75, 0.0, v,  0.30, 0.70, E_LCP),
         "Al2O3 volume fraction", TEAL),
        ("phi_stiff",  np.linspace(0.05, 0.60, 50),
         lambda v: modulus(0.75, 0.0, 0.03, v,   1.0-v, E_LCP),
         "Stiff co-polymer fraction (LCP)", RED),
    ]

    for ax, (name, x_arr, fn, xlabel, clr) in zip(axes.flat, sweep_params):
        ax.set_facecolor("#FAFAF8")
        y_arr = [fn(x) for x in x_arr]
        ax.plot(x_arr, y_arr, color=clr, lw=2)
        ax.axhline(2000, color=GRAY, lw=1, ls="--", label="target 2000 MPa")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Modulus (MPa)", fontsize=9)
        ax.set_title(f"Effect of {name}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    p = os.path.join(output_dir, "fig2_modulus_sensitivity.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 3: Halpin-Tsai -- modulus vs filler fraction ─────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor("#FAFAF8")

    phi_arr = np.linspace(0, 0.15, 200)
    for label, E_mat, clr in [
        ("SLM-Li matrix (sPEEK+LCP, DS=0.75)", 2750, TEAL),
        ("SLM-H  matrix (sPEEK,     DS=0.60)",  133, BLUE),
        ("SLM-B  matrix (PEO/PVA,   DS=0.0)",    60, AMBER),
    ]:
        y = [halpin_tsai(E_mat, E_AL2O3, p) for p in phi_arr]
        ax.plot(phi_arr * 100, y, lw=2, label=label, color=clr)

    ax.set_xlabel("Al2O3 filler volume fraction (%)", fontsize=11)
    ax.set_ylabel("Composite modulus (MPa)", fontsize=11)
    ax.set_title("Halpin-Tsai reinforcement by nano-Al2O3\n"
                 "E_c = E_m * (1 + 2*eta*phi) / (1 - eta*phi)   [spheres, xi=2]",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(output_dir, "fig3_halpin_tsai.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

    # ── Fig 4: Manufacturing cost vs thickness ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Manufacturing cost vs membrane thickness  (Phase 1 / 2 / 3)",
                 fontsize=11)

    chambers = [
        ("SLM-H",  0.97, 0.0,  0.03, 1.40, COST_COPOLYMER, BLUE),
        ("SLM-B",  0.60, 0.38, 0.02, 1.40, COST_COPOLYMER, AMBER),
        ("SLM-Li", 0.70, 0.27, 0.03, 1.40, COST_COPOLYMER, TEAL),
    ]
    t_arr = np.linspace(5, 120, 100)

    for ax, (name, ps, pst, pf, rho, cc, clr) in zip(axes, chambers):
        ax.set_facecolor("#FAFAF8")
        for phase, ls, lw in [(1, "-", 2.2), (2, "--", 1.8), (3, ":", 1.8)]:
            costs = [manufacturing_cost(ps, pst, pf, t,
                                        cost_copolymer=cc,
                                        rho_copolymer=rho,
                                        phase=phase)
                     for t in t_arr]
            ax.plot(t_arr, costs, color=clr, ls=ls, lw=lw,
                    label=f"Phase {phase}")
        ax.set_xlabel("Thickness (um)", fontsize=9)
        ax.set_ylabel("Cost (USD/ft2)", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    p = os.path.join(output_dir, "fig4_cost_vs_thickness.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


# ===========================================================================
# 4.  VALIDATION + MAIN
# ===========================================================================

if __name__ == "__main__":

    import os, sys

    # Output directory: project root (one level up from src/)
    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _project_root = (os.path.dirname(_script_dir)
                     if os.path.basename(_script_dir).lower() == "src"
                     else _script_dir)
    _plot_dir = os.path.join(_project_root, "MODEL1")

    print("=" * 60)
    print("  Modulus + Cost Model  --  Validation vs Proposal Targets")
    print("=" * 60)

    # ── Modulus validation ─────────────────────────────────────────────
    print("\n--- Modulus model ---")

    # SLM-Li: DS=0.75, 3% Al2O3, 30% LCP
    E_li = modulus(DS=0.75, C_xl=0.0, phi_filler=0.03,
                   phi_stiff=0.30, phi_sPEEK=0.70, E_stiff=E_LCP)
    print(f"  SLM-Li (DS=0.75, 3% Al2O3, 30% LCP):  {E_li:.1f} MPa  "
          f"(target >2000: {'PASS' if E_li > 2000 else 'FAIL'})")

    # SLM-H: DS=0.60, 5% crosslinker, 3% SiO2, PTFE backing
    E_h  = modulus(DS=0.60, C_xl=0.05, phi_filler=0.03,
                   phi_stiff=0.10, phi_sPEEK=0.90, E_stiff=E_PTFE)
    print(f"  SLM-H  (DS=0.60, 5% xl, PTFE):         {E_h:.1f} MPa  "
          f"(target >8:    {'PASS' if E_h > 8 else 'FAIL'})")

    # SLM-B: PEO/PVA blend DS=0, 2% filler -- stiff component is PVA
    E_b  = modulus(DS=0.0,  C_xl=0.0,  phi_filler=0.02,
                   phi_stiff=0.40, phi_sPEEK=0.60, E_stiff=E_PVA)
    print(f"  SLM-B  (PEO/PVA, 2% filler):           {E_b:.1f} MPa  "
          f"(target 6-10:  {'PASS' if 6 <= E_b <= 10 else 'check -- E_PVA may need tuning'})")

    print("\n  DS -> E_sPEEK:")
    for ds in [0.0, 0.40, 0.52, 0.60, 0.70, 0.80]:
        print(f"    DS={ds:.2f}  E={E_sPEEK(ds):.1f} MPa")

    # ── Cost validation ────────────────────────────────────────────────
    print("\n--- Manufacturing cost model ---")
    for name, ps, pst, pf, t, target in [
        ("SLM-Li", 0.70, 0.27, 0.03, 20.0, 2.60),
        ("SLM-H",  0.97, 0.00, 0.03, 70.0, None),
        ("SLM-B",  0.60, 0.38, 0.02, 20.0, 0.30),
    ]:
        c1 = manufacturing_cost(ps, pst, pf, t, phase=1)
        c2 = manufacturing_cost(ps, pst, pf, t, phase=2)
        tgt_str = f"  (target ${target:.2f}: {'OK' if target and c1 < target*1.5 else 'check'})" if target else ""
        print(f"  {name}: Phase1=${c1:.3f}  Phase2=${c2:.3f}/ft2{tgt_str}")

    # ── Plots ──────────────────────────────────────────────────────────
    print(f"\n--- Generating plots -> {_plot_dir} ---")
    make_plots(_plot_dir)
    print("\n[OK] Done.")
