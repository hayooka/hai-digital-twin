"""Verbatim limitation text from the Student Guide §5.1-§5.6."""

LIMITATIONS = [
    {
        "id": "5.1",
        "title": "It cannot replace the real testbed",
        "body": (
            "The surrogate is a learned approximation, not a physics simulation. "
            "It does not encode conservation laws, thermodynamic equations, or fluid "
            "dynamics. It reproduces patterns observed during 194 hours of training. "
            "Any behavior outside that envelope (pressures above 2 bar, temperatures "
            "above 35°C, tank levels near the physical limits) is extrapolation, and "
            "the model's output in those regimes is unreliable."
        ),
    },
    {
        "id": "5.2",
        "title": "It cannot discover new attack types",
        "body": (
            "The surrogate propagates perturbations you define. It does not know what "
            "constitutes a realistic attack. If you inject a +500mm bias on the LC "
            "setpoint (well beyond the 300-500mm normal range), the surrogate will "
            "produce numbers, but those numbers represent the linear extrapolation "
            "of a neural network, not the response of a real water tank being "
            "overfilled. The attack taxonomy comes from you, not from the model."
        ),
    },
    {
        "id": "5.3",
        "title": "It cannot simulate internal DCS attacks",
        "body": (
            "The HAI dataset includes 8 internal-point attack scenarios (AE01-AE08) "
            "that manipulate algorithm block parameters inside the Emerson Ovation "
            "DCS. These attacks modify things like PID gain calibration curves and "
            "alarm thresholds. The surrogate uses HAIEnd columns as plant inputs but "
            "does not predict them as outputs. You cannot inject a perturbation into "
            "a HAIEnd internal variable and observe its effect."
        ),
    },
    {
        "id": "5.4",
        "title": "It cannot model P2, P3, or P4",
        "body": (
            "The surrogate covers P1 (boiler) only. The turbine process (P2, GE "
            "Mark VIe DCS), water treatment process (P3, Siemens S7-300 PLC), and "
            "HIL simulation (P4, dSPACE SCALEXIO) are not modeled. Cross-process "
            "coupling through P4 is not captured. Attacks targeting P2, P3, or P4 "
            "signals cannot be simulated."
        ),
    },
    {
        "id": "5.5",
        "title": "It cannot produce publication-grade synthetic datasets",
        "body": (
            "Reviewers should never be told that synthetic data from this surrogate "
            "is equivalent to real testbed data. Three of five loops fail the KS "
            "distributional fidelity test (LC 0.215, TC 0.157, CC 0.121). The CC "
            "controller achieves only 77% classifier accuracy. Any experimental "
            "results obtained on synthetic data must carry explicit caveats about "
            "the surrogate's limitations and the KS statistics for each loop used."
        ),
    },
    {
        "id": "5.6",
        "title": "It cannot guarantee attack response fidelity",
        "body": (
            "The closed-loop NRMSE (Gate 3) was measured on normal operation only. "
            "There is no ground truth for attack response accuracy because you "
            "cannot run the same attack on the real testbed and compare. The attack "
            "responses are plausible (correct direction, reasonable magnitude, "
            "physical clamping) but not validated against real attack data. Treat "
            "them as qualitative simulations, not quantitative predictions."
        ),
    },
]

CAPABILITIES_SHORT = [
    "Parameterize new attacks (what happens if magnitude doubles, loops combine, etc.)",
    "Stress-test anomaly detectors on synthetic normal + attack data",
    "Map loop dynamics: response delays, cross-loop coupling, scale relationships",
    "Per-scenario conditioned forecasting (Normal / AP_no / AP_with / AE_no)",
]

# Per-loop KS from guide §3.3 (distributional fidelity on tracking error)
KS_PER_LOOP_REPORTED = {
    "PC": {"ks": 0.060, "status": "PASS"},
    "LC": {"ks": 0.215, "status": "FAIL"},
    "FC": {"ks": 0.067, "status": "PASS"},
    "TC": {"ks": 0.157, "status": "FAIL"},
    "CC": {"ks": 0.121, "status": "FAIL"},
}
