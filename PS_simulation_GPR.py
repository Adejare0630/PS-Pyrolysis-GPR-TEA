#%% [markdown]
# PS pyrolysis recycling - BioSTEAM process model with Gaussian-process yields, TEA and LCA
#
# What changed from Simulation_october_11.py
#   * Reactor yields come from the Gaussian process (PS_gaussian_process.joblib) instead of the DeepNN.
#     Inputs: temperature, regime class (continuous) and catalyst class (ZnO = "basic").
#     Liquid, styrene, BTX and total aromatics are predicted; the "new-process" prediction (averaged over
#     studies, with between-study variance) is used, and styrene + BTX <= total aromatics <= liquid is enforced.
#   * MSPs are reported for styrene and BTX individually, two ways: (a) market-value allocation - all product
#     prices scaled by one factor until NPV = 0 (MSP_i = factor x market price_i); (b) one product's price solved
#     with the other products at market price (as in the original). Plus the mean liquid-product MSP.
#   * GP uncertainty is carried to the minimum selling price (MSP) and GWP by Monte Carlo
#     (GP standard deviation widened 1.3x because the raw 95% intervals cover only ~88-92% of runs).
#   * Separate chemicals for the polymer (PS) and the monomer (Styrene); light gases are hydrocarbons
#     (PS contains no oxygen, so the old 90 % CO2 gas split is removed).
#   * Vapour coolers are solved rigorously (VLE), so condensation happens in the coolers; the old non-rigorous
#     coolers left the stream as vapour and reported a wrong outlet temperature.
#   * Figures include a process flow diagram with numbered streams and a stream table, and mass/energy Sankeys.
#   * Reactor heat duty is calculated (sensible + vaporisation + depolymerisation enthalpy) and supplied by a
#     natural-gas fired heater; char on the catalyst is burnt in the regenerator and its heat is credited.
#   * Bug fixes: screen recycle now re-enters the feed (Feed_handling dropped its second inlet); labour cost now enters the fixed operating cost; equipment costs divided by 1e6 (was 1**6);
#     natural gas impact uses MJ = kg x LHV (factor is per MJ; the old key never matched); electricity impact
#     uses the plant's net electricity; product impact factors are mapped to the rows in impact_factors.csv
#     (old product IDs never matched); BTX factor no longer re-uses the aromatics factor; reactor-temperature
#     sensitivity is +/- 50 C inside the GP data range instead of +/-20 % in Kelvin (High/Low were swapped).
#
# Folder layout expected (same as before):
#   <folder>/PS_simulation_GPR.py
#   <folder>/PS_gaussian_process.joblib         (or the training table -> the GP is retrained automatically)
#   <folder>/Module/  _feed_handling.py, _grinder.py, _chscreen.py, _tea.py, impact_factors.csv
#   (Module/ and impact_factors.csv are also found one level up, e.g. Downloads/Module)
#
# Run cell by cell in VS Code, or:  python PS_simulation_GPR.py
# Outputs go to <folder>/simulation_outputs/ (figures + PS_simulation_GPR_results.xlsx)
# Sankey diagrams need plotly (pip install plotly); PNG export of them also needs kaleido (pip install kaleido)

#%% Imports and paths
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
import biosteam as bst
import thermosteam as tmo
from biosteam.units.decorators import cost

warnings.filterwarnings("ignore")
# folder of this script, also in the VS Code interactive window / Jupyter (where __file__ is not defined and the
# working folder can be a different one, e.g. Downloads, holding older data or model files)
HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or globals().get("__vsc_ipynb_file__")
                                       or os.path.join(os.getcwd(), "_")))
SEARCH = [HERE, os.path.join(HERE, "Module"), os.path.join(HERE, "..", "Module"),
          os.path.join(HERE, "..", "PS Journals for dataset")]
for p in SEARCH[1:3]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def find_file(*names):
    for folder in SEARCH:
        for n in names:
            f = os.path.join(folder, n)
            if os.path.isfile(f):
                return os.path.abspath(f)
    raise FileNotFoundError(f"None of {names} found in {SEARCH}")


from _feed_handling import Feed_handling      # user modules (unchanged)
from _grinder import Grinder
from _chscreen import Screen
from _tea import TEA_PS

OUT = os.path.join(HERE, "simulation_outputs")
os.makedirs(OUT, exist_ok=True)
# Figures are drawn at the size they are printed (at most ~7 in wide, a full journal page width),
# so the font sizes below are the printed sizes (8-10 pt) and nothing needs to be shrunk in Word.
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "gold": "#eda100", "grey": "#8a8a8a"}

bst.settings.CEPCI = 800                         # CEPCI 2024
bst.settings.electricity_price = 0.0813           # $/kWh (EIA 2024)
bst.preferences.update(flow="kg/hr", T="degK", P="Pa", N=100, composition=False)

#%% Prices and process assumptions (edit here)
prices = {
    "PS": 0.347,                   # $/kg waste PS feed: scrap PS, Intratec polystyrene prices ($347/t)
                                   # https://www.intratec.us/solutions/primary-commodity-prices/commodity/polystyrene-prices
                                   # (previously 0.90 = businessanalytiq virgin-PS index; now a pessimistic scenario)
    "Styrene": 0.96,               # $/kg (businessanalytiq styrene index)
    "BTX": 0.97 * 0.8,             # $/kg (toluene index x 0.8)
    "Aromatics": 1.08 * 0.8,       # $/kg (ethylbenzene, Intratec x 0.8)
    "NG": 7.40 * 1000 * 1.525 / 28316.8,   # $/kg natural gas
    "Fresh catalyst": 1.09,        # $/kg ZnO (Intratec)
    "N2": 0.10,                    # $/kg bulk nitrogen for the fluidising gas (assumed; once-through, conservative)
}
CEPCI_2015 = 556.8                 # cost index of the NREL 2015 reactor/regenerator correlations
A = dict(
    capacity_tpd=250,              # PS feed, tonnes/day
    T_C=500,                       # baseline reactor temperature (°C); ZnO/continuous data cover 430-510 °C
    regime="continuous",           # GP regime class (fluidised bed)
    catalyst="basic",              # GP catalyst class: ZnO is a basic metal oxide; "none" = thermal pyrolysis
    gp_sd_scale=1.3,               # widen GP sd for Monte Carlo (calibration diagnostic)
    max_liquid=99.0,               # wt%; keeps >= 1 wt% gas + char (GP draws can reach 100)
    solid_share_nonliquid=0.50,    # char share of (100 - liquid); median of runs reporting gas and solid
    btx_split={"Benzene": 0.14, "Toluene": 0.40, "Ethylbenzene": 0.46},   # EB share = data median
    gas_split={"CH4": 0.25, "C2H4": 0.30, "C2H6": 0.10, "C3H6": 0.30, "H2": 0.05},  # mass fractions
    catalyst_to_feed=0.05,         # circulating ZnO : PS (wt)
    catalyst_purge=0.02,           # fraction of circulating catalyst purged and replaced
    fluid_gas_to_feed=0.20,        # N2 make-up for the fluidising-gas loop (kg/kg PS)
    dH_depolymerisation=670.,      # kJ/kg PS (styrene polymerisation enthalpy, ~70 kJ/mol)
    furnace_efficiency=0.80,       # fired-heater efficiency (LHV)
    regen_heat_recovery=0.80,      # fraction of char combustion heat returned to the reactor
    NG_LHV=50.0,                   # MJ/kg
    GWP_CH4=25.,                   # kg CO2e/kg CH4 (as in the original model)
)
GP_DATA_RANGE = {("continuous", "basic"): (430, 510), ("continuous", "none"): (330, 780)}

#%% Chemicals
def create_chemicals():
    def chem(ID, search, **kw):
        c = tmo.Chemical(ID, search_ID=search, **kw)
        return c
    styrene = chem("Styrene", "Styrene")
    PS = styrene.copy("PS")                       # polymer, represented with styrene properties; fully converted
    chems = [PS, styrene,
             chem("Benzene", "Benzene"), chem("Toluene", "Toluene"), chem("Ethylbenzene", "Ethylbenzene"),
             chem("AMS", "alpha-Methylstyrene"),  # other mono-aromatics (alpha-methylstyrene surrogate)
             chem("Heavies", "1081-75-0"),        # dimers/trimers & unidentified liquid (1,3-diphenylpropane)
             chem("CH4", "Methane"), chem("C2H4", "Ethylene"), chem("C2H6", "Ethane"),
             chem("C3H6", "Propylene"), chem("H2", "Hydrogen"),
             chem("N2", "N2"), chem("O2", "O2"), chem("CO2", "CO2"), chem("H2O", "Water"),
             chem("ZnO", "ZnO", phase="s"), chem("Char", "C", phase="s"),
             chem("Ash", "CaO", phase="s")]            # needed by the boiler's combustion reactions
    chems = tmo.Chemicals(chems)
    for c in chems:
        c.default()
    chems["ZnO"].Hf = 0; chems["ZnO"].HHV = 0; chems["ZnO"].LHV = 0
    chems = tmo.CompiledChemicals(chems)
    chems.set_synonym("H2O", "Water")
    chems.set_synonym("CH4", "Methane")
    return chems

chemicals = create_chemicals()
bst.settings.set_thermo(chemicals)

#%% Gaussian-process yield model
class GPYieldModel:
    """Loads the trained GPRs (or retrains them) and returns consistent yields in wt% of PS feed."""
    targets = ["Liquid", "Styrene", "BTX", "Total aromatics"]

    def __init__(self):
        try:
            _gp_file = find_file("PS_gaussian_process.joblib"); print("GP model loaded from", _gp_file)
            b = joblib.load(_gp_file)
            self.models, self.R, self.C, self.S = b["models"], b["regimes"], b["catalysts"], b["studies"]
            print("GP loaded from PS_gaussian_process.joblib")
        except Exception as e:                    # missing file or scikit-learn version mismatch
            print(f"Could not load the saved GP ({type(e).__name__}); retraining from the training table ...")
            self._train()
        self._cache = {}

    def _train(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
        cols = {"Liquid": "Liquid (wt% of feed)", "Styrene": "Styrene (wt% of feed)",
                "BTX": "BTX (wt% of feed)", "Total aromatics": "Total aromatics (wt% of feed)"}
        df = pd.read_excel(find_file("PS_ML_training_table_1.xlsx", "PS_ML_training_table.xlsx"),
                           sheet_name="Training table", header=1)
        df = df.rename(columns={"Temperature (°C)": "T", "Regime class": "regime",
                                "Catalyst class": "catalyst", "Study ID (group for CV)": "study_id"})
        df["T"] = pd.to_numeric(df["T"], errors="coerce")
        excl = (df["Feed description"].astype(str).str.contains("HIPS")
                | (pd.to_numeric(df["PS (wt% of feed)"], errors="coerce") < 100)
                | df["Pressure as reported"].astype(str).str.fullmatch("Vacuum"))
        df = df[~excl]
        self.R = sorted(df["regime"].dropna().unique()); self.C = sorted(df["catalyst"].dropna().unique())
        self.S = sorted(df["study_id"].dropna().unique())
        self.models = {}
        for name, col in cols.items():
            d = df.assign(y=pd.to_numeric(df[col], errors="coerce")).dropna(subset=["T", "regime", "catalyst", "y"])
            X = self._X(d["T"].values, d["regime"].values, d["catalyst"].values, d["study_id"].values)
            k = (Ck(1.0, (1e-3, 1e3)) * RBF([3.0] * X.shape[1], (1.5, 1e4)) + WhiteKernel(0.3, (1e-4, 10)))
            self.models[name] = make_pipeline(StandardScaler(), GaussianProcessRegressor(
                k, normalize_y=True, n_restarts_optimizer=3, random_state=42)).fit(X, d["y"].values)
        joblib.dump({"models": self.models, "regimes": self.R, "catalysts": self.C, "studies": self.S,
                     "targets": self.targets}, os.path.join(HERE, "PS_gaussian_process.joblib"))

    def _X(self, T, reg, cat, study):
        T = np.atleast_1d(T).astype(float)
        cols = [T] + [np.asarray(reg) == r for r in self.R] + [np.asarray(cat) == c for c in self.C] \
               + [np.asarray(study) == s for s in self.S]
        return np.column_stack([np.broadcast_to(c, T.shape).astype(float) for c in cols])

    def raw(self, T_C, regime, catalyst):
        """New-process mean and sd (wt% of feed): average over studies, sd includes between-study variance."""
        key = (round(float(T_C), 3), regime, catalyst)
        if key not in self._cache:
            n = len(self.S)
            X = self._X(np.full(n, T_C), [regime] * n, [catalyst] * n, self.S)
            mean, sd = {}, {}
            for name, m in self.models.items():
                mu, s = m[-1].predict(m[0].transform(X), return_std=True)
                mean[name] = float(mu.mean()); sd[name] = float(np.sqrt((s ** 2).mean() + mu.var()))
            self._cache[key] = (mean, sd)
        return self._cache[key]

    @staticmethod
    def consistent(y):
        """Clip to 0-100, cap liquid at max_liquid and enforce styrene + BTX <= total aromatics <= liquid."""
        y = {k: float(np.clip(v, 0, 100)) for k, v in y.items()}
        y["Liquid"] = min(y["Liquid"], A["max_liquid"])
        y["Total aromatics"] = min(y["Total aromatics"], y["Liquid"])
        sb = y["Styrene"] + y["BTX"]
        if sb > y["Total aromatics"]:            # scale styrene and BTX down together
            f = y["Total aromatics"] / sb
            y["Styrene"] *= f; y["BTX"] *= f
        return y

    def mean(self, T_C, regime, catalyst):
        return self.consistent(self.raw(T_C, regime, catalyst)[0])

    # Correlation of the GP prediction errors between outputs (5-fold out-of-fold residuals, 140 complete-case
    # runs from 19 studies; order: Liquid, Styrene, BTX, Total aromatics). Without it the Monte Carlo would
    # treat, e.g., liquid and total aromatics as independent although their errors are strongly linked.
    error_corr = np.array([[1.00, 0.01, -0.05, 0.80],
                           [0.01, 1.00, -0.45, 0.10],
                           [-0.05, -0.45, 1.00, 0.11],
                           [0.80, 0.10, 0.11, 1.00]])

    def sample(self, T_C, regime, catalyst, rng, sd_scale=1.0):
        mean, sd = self.raw(T_C, regime, catalyst)
        mu = np.array([mean[k] for k in self.targets]); s = sd_scale * np.array([sd[k] for k in self.targets])
        draw = rng.multivariate_normal(mu, self.error_corr * np.outer(s, s))
        return self.consistent(dict(zip(self.targets, draw)))

    def interval(self, T_C, regime, catalyst, z=1.96):
        mean, sd = self.raw(T_C, regime, catalyst)
        return {k: (max(0, mean[k] - z * sd[k]), min(100, mean[k] + z * sd[k])) for k in mean}

gp = GPYieldModel()
print(pd.DataFrame({T: gp.mean(T, A["regime"], A["catalyst"]) for T in (450, 475, 500)}).round(1))

#%% Unit operations
class GPPyrolyzer(bst.Unit):
    """Fluidised-bed pyrolyzer; yields from the GP, heat supplied by a natural-gas fired heater.
    ins : [0] PS + catalyst + fluidising gas, [1] natural gas, [2] combustion air
    outs: [0] hot vapours + solids, [1] flue gas"""
    _N_ins = 3
    _N_outs = 2
    _F_BM_default = {"Reactor": 2.3}
    _units = {"Feed": "lb/hr", "Duty": "kJ/hr"}

    def _init(self, T=773.15, regime="continuous", catalyst="basic", yields=None):
        self.T = T; self.regime = regime; self.catalyst = catalyst
        self.yields = yields               # dict to override the GP mean (Monte Carlo / sensitivity)
        self.heat_credit = 0.              # kJ/hr from char combustion in the regenerator
        self.last_yields = None

    def _run(self):
        feed, fuel, air = self.ins
        out, flue = self.outs
        y = self.yields or gp.mean(self.T - 273.15, self.regime, self.catalyst)
        self.last_yields = y
        F = feed.imass["PS"]
        out.copy_like(feed); out.imass["PS"] = 0
        L, S, B, TA = (y[k] / 100 for k in ("Liquid", "Styrene", "BTX", "Total aromatics"))
        out.imass["Styrene"] += S * F
        for c, f in A["btx_split"].items():
            out.imass[c] += f * B * F
        out.imass["AMS"] += (TA - S - B) * F
        out.imass["Heavies"] += (L - TA) * F
        # numerical floor: keep every liquid species >= 0.01 wt% so no column receives an empty key
        # (taken from the gas, so the reactor mass balance still closes)
        floor = 0.
        for c in ("Styrene", "Benzene", "Toluene", "Ethylbenzene", "AMS", "Heavies"):
            add = max(1e-4 * F - out.imass[c], 0); out.imass[c] += add; floor += add
        solid = A["solid_share_nonliquid"] * (1 - L)
        out.imass["Char"] += solid * F
        gas = (1 - L - solid) * F - floor
        for c, f in A["gas_split"].items():
            out.imass[c] += f * gas
        out.T = self.T; out.P = feed.P; out.phase = "g"
        # energy balance: sensible + latent (stream enthalpies) + depolymerisation enthalpy
        self.duty = out.H - feed.H + A["dH_depolymerisation"] * F
        Q_fuel = max(self.duty - self.heat_credit, 0) / A["furnace_efficiency"]
        fuel.empty(); air.empty(); flue.empty()
        fuel.imass["CH4"] = Q_fuel / (A["NG_LHV"] * 1000)
        n = fuel.imol["CH4"]
        air.imol["O2"] = 2 * n * 1.15; air.imol["N2"] = air.imol["O2"] * 79 / 21
        flue.imol["CO2"] = n; flue.imol["H2O"] = 2 * n
        flue.imol["O2"] = air.imol["O2"] - 2 * n; flue.imol["N2"] = air.imol["N2"]
        flue.T = 450.; flue.phase = "g"

    def _design(self):
        self.design_results["Feed"] = self.ins[0].get_total_flow("lb/hr")
        self.design_results["Duty"] = self.duty

    def _cost(self):  # NREL 2015 (fy15osti/62455), escalated from the 2015 CEPCI to bst.settings.CEPCI
        self.baseline_purchase_costs["Reactor"] = (2 * 3818000 * (self.design_results["Feed"] / 2526000) ** 0.5
                                                   * bst.settings.CEPCI / CEPCI_2015)


@cost(basis="Flow", ID="Cyclone", units="tonnes/day", S=500, CE=567.5, cost=982510.7, n=0.6, BM=4.28)
class SolidsCyclone(bst.Unit):
    """Removes catalyst and char from the hot vapours (outs: [0] vapours, [1] solids)."""
    _N_ins = 1; _N_outs = 2
    _units = {"Flow": "tonnes/day"}
    solids = ("ZnO", "Char")

    def _run(self):
        feed = self.ins[0]; vap, sol = self.outs
        vap.copy_like(feed); sol.empty()
        for s in self.solids:
            sol.imass[s] = feed.imass[s]; vap.imass[s] = 0
        sol.T = vap.T = feed.T; sol.phase = "s"; vap.phase = "g"

    def _design(self):
        self.design_results["Flow"] = self.ins[0].get_total_flow("tonnes/day")


class CatalystRegenerator(bst.Unit):
    """Burns char off the spent catalyst with 20 % excess air; returns regenerated ZnO and purges a fraction.
    ins : [0] spent solids, [1] air       outs: [0] regenerated catalyst, [1] flue gas, [2] purged catalyst"""
    _N_ins = 2; _N_outs = 3
    _F_BM_default = {"Regenerator": 2.3}
    _units = {"Feed": "lb/hr"}

    def _init(self, purge=0.02, T=773.15, pyrolyzer=None):
        self.purge = purge; self.T = T; self.pyrolyzer = pyrolyzer

    def _run(self):
        solids, air = self.ins; regen, flue, purge = self.outs
        nC = solids.imol["Char"]
        air.empty(); air.imol["O2"] = 1.2 * nC; air.imol["N2"] = air.imol["O2"] * 79 / 21
        flue.empty(); flue.imol["CO2"] = nC; flue.imol["O2"] = 0.2 * nC; flue.imol["N2"] = air.imol["N2"]
        flue.T = self.T; flue.phase = "g"
        zno = solids.imass["ZnO"]
        regen.empty(); purge.empty()
        regen.imass["ZnO"] = zno * (1 - self.purge); purge.imass["ZnO"] = zno * self.purge
        regen.T = purge.T = self.T; regen.phase = purge.phase = "s"
        self.heat_released = solids.imass["Char"] * 32800 * A["regen_heat_recovery"]   # kJ/hr
        if self.pyrolyzer is not None:
            self.pyrolyzer.heat_credit = self.heat_released

    def _design(self):
        self.design_results["Feed"] = self.ins[0].get_total_flow("lb/hr") + self.ins[1].get_total_flow("lb/hr")

    def _cost(self):  # same fluidised-vessel correlation as the pyrolyzer (escalated to bst.settings.CEPCI)
        self.baseline_purchase_costs["Regenerator"] = (3818000 * (self.design_results["Feed"] / 2526000) ** 0.5
                                                       * bst.settings.CEPCI / CEPCI_2015)

#%% Flowsheet
def create_ps_sys(T_C=None, catalyst=None, regime=None, capacity_tpd=None, yields=None, ID="PS_sys"):
    """Build the flowsheet; arguments left as None take the current values in A."""
    T_C = A["T_C"] if T_C is None else T_C
    catalyst = A["catalyst"] if catalyst is None else catalyst
    regime = A["regime"] if regime is None else regime
    capacity_tpd = A["capacity_tpd"] if capacity_tpd is None else capacity_tpd
    bst.main_flowsheet.set_flowsheet(ID)
    bst.main_flowsheet.clear()
    bst.settings.set_thermo(chemicals)
    T = T_C + 273.15
    feed = bst.Stream("PS_Feed", PS=capacity_tpd * 1000 / 24, units="kg/hr", T=298.15, price=prices["PS"])
    F = feed.F_mass
    circ = A["catalyst_to_feed"] * F if catalyst != "none" else 0.
    makeup = bst.Stream("catalyst", ZnO=circ * A["catalyst_purge"], units="kg/hr", phase="s",
                        price=prices["Fresh catalyst"])
    fluid_gas = bst.Stream("fluid_gas", N2=A["fluid_gas_to_feed"] * F, units="kg/hr", price=prices["N2"])
    pyro_ng = bst.Stream("pyrolyzer_ng", price=prices["NG"])
    pyro_air = bst.Stream("pyrolyzer_air")
    regen_air = bst.Stream("regenerator_air")
    regenerated = bst.Stream("regenerated_catalyst", ZnO=circ * (1 - A["catalyst_purge"]),
                             units="kg/hr", phase="s", T=T)      # initial guess for the recycle
    recycle = bst.Stream("screen_recycle")

    # pretreatment
    feed_mix = bst.Mixer("Feed_mixer", ins=[feed, recycle])     # Feed_handling only passes its first inlet
    handling = Feed_handling("Handling", ins=[feed_mix - 0, bst.Stream("handling_spare")])
    grinder = Grinder("Grinder", ins=handling - 0)
    screen = Screen("CHScreen", ins=grinder - 0, outs=("CRPS", recycle))
    mix = bst.Mixer("Mixer_pyrolyzer", ins=[screen - 0, makeup, regenerated, fluid_gas])
    # pyrolysis + catalyst loop
    reactor = GPPyrolyzer("Pyrolyzer", ins=[mix - 0, pyro_ng, pyro_air], outs=("PyrolyzerO", "pyrolyzer_flue"),
                          T=T, regime=regime, catalyst=catalyst, yields=yields)
    cyclone = SolidsCyclone("Cyclone", ins=reactor - 0, outs=("oil_gas", "spent_solids"))
    CatalystRegenerator("Regenerator", ins=[cyclone - 1, regen_air],
                        outs=(regenerated, "regenerator_flue", "purged_catalyst"),
                        purge=A["catalyst_purge"], T=T, pyrolyzer=reactor)
    # vapour recovery
    cooler1 = bst.HXutility("Cooler_CW", ins=cyclone - 0, T=320.0, rigorous=True)      # cooling water
    cooler2 = bst.HXutility("Cooler_chilled", ins=cooler1 - 0, T=283.15, rigorous=True)   # chilled water/brine
    cond = bst.Flash("Condenser", ins=cooler2 - 0, outs=("offgas", "pyrolysis_oil"), T=283.15, P=101325)
    pump = bst.Pump("Pump1", ins=cond - 1, P=2 * 101325)
    # product separation
    D1 = bst.BinaryDistillation("DBTX", ins=pump - 0, outs=("BTX", ""), LHK=("Ethylbenzene", "Styrene"),
                                product_specification_format="Recovery", Lr=0.95, Hr=0.99, k=1.5,
                                P=101325, partial_condenser=False)          # EB/styrene splitter
    D2 = bst.BinaryDistillation("DStyrene", ins=D1 - 1, outs=("Styrene", ""), LHK=("Styrene", "AMS"),
                                product_specification_format="Recovery", Lr=0.99, Hr=0.99, k=1.5,
                                P=101325, partial_condenser=False)
    D3 = bst.BinaryDistillation("DAromatics", ins=D2 - 1, outs=("Aromatics", "Bottoms"), LHK=("AMS", "Heavies"),
                                product_specification_format="Recovery", Lr=0.95, Hr=0.95, k=1.5,
                                P=101325, partial_condenser=False)
    D1.outs[0].price = prices["BTX"]; D2.outs[0].price = prices["Styrene"]
    D3.outs[0].price = prices["Aromatics"]
    # boiler-turbogenerator burns off-gas + heavy bottoms, buys NG only if needed
    bst.BoilerTurbogenerator("Turbogenerator", ins=(D3 - 1, cond - 0, "boiler_makeup_water",
                             "boiler_turbogenerator_ng", "boiler_lime", "boiler_chemicals", "boiler_air"),
                             boiler_efficiency=0.85, turbogenerator_efficiency=0.85,
                             natural_gas_price=prices["NG"])
    system = bst.main_flowsheet.create_system(ID)
    system.set_tolerance(mol=1e-3, rmol=1e-4, maxiter=200)
    return system


def S(system, ID):
    return system.flowsheet.stream[ID]


def U(system, ID):
    return system.flowsheet.unit[ID]

#%% TEA
employee_costs = {  # salary, staff (Dutta 2002, BLS-adjusted); scaled linearly from a 2000 t/d plant
    "Plant Manager": [159000, 1], "Plant Engineer": [94000, 1], "Maintenance Supr": [87000, 1],
    "Maintenance Tech": [62000, 6], "Lab Manager": [80000, 1], "Lab Technician": [58000, 1],
    "Shift Supervisor": [80000, 3], "Shift Operators": [62000, 12], "Yard Employees": [36000, 4],
    "Clerks & Secretaries": [43000, 1], "General Manager": [188000, 0]}
PRODUCTS = ["BTX", "Styrene", "Aromatics"]


class PSTEA(TEA_PS):
    """TEA_PS with labour, fringe benefits and supplies added to the fixed operating cost
    (TEA_PS stored labor_cost but never used it), and the boiler-turbogenerator installed with its own
    bare-module factors (BioSTEAM/Humbird, ~1.8) instead of the plant-wide Lang factor."""
    def _FOC(self, FCI):
        return super()._FOC(FCI) + self.labor_cost * (1 + self.fringe_benefits + self.supplies)

    def _DPI(self, installed_equipment_cost):
        if self.lang_factor:
            for u in self.system.cost_units:
                if isinstance(u, bst.BoilerTurbogenerator):
                    installed_equipment_cost += u.installed_cost - u.purchase_cost * self.lang_factor
        return installed_equipment_cost


LABOR_SCALING = 0.25               # staff cost ~ capacity^0.25 (a plant needs a minimum crew whatever its size)


def get_tea(system, lang_factor=5.05, IRR=0.10):
    F = S(system, "PS_Feed").F_mass
    labor = sum(s * n for s, n in employee_costs.values()) * (F / (2000 * 1000 / 24)) ** LABOR_SCALING
    return PSTEA(system=system, IRR=IRR, duration=(2020, 2040), depreciation="MACRS7", income_tax=0.21,
                 operating_days=350, lang_factor=lang_factor, construction_schedule=(0.4, 0.6),
                 WC_over_FCI=0.05, labor_cost=labor, fringe_benefits=0.4, property_tax=0.001,
                 property_insurance=0.005, supplies=0.20, maintenance=0.004, administration=0.005,
                 finance_fraction=0.4, finance_years=10, finance_interest=0.07)


def product_flow_tpy(tea):
    return sum(S(tea.system, p).get_total_flow("tonne/year") for p in PRODUCTS)


MSP_PRODUCTS = ["Styrene", "BTX"]          # products whose individual MSPs are reported


def _solve_scale(tea, weights):
    """Find x so that NPV = 0 when each product price = x * weight (secant; NPV is ~linear in x)."""
    streams = [S(tea.system, p) for p in PRODUCTS]
    old = [s.price for s in streams]

    def npv(x):
        for s, w in zip(streams, weights): s.price = x * w
        return tea.NPV
    a, b = 0.0, 1.0
    fa, fb = npv(a), npv(b)
    for _ in range(30):
        if fb == fa:                                   # flat secant: cannot improve further
            break
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b, fb = b, fb, c, npv(c)
        if abs(fb) < 1: break
    for s, o in zip(streams, old): s.price = o
    return b


def mean_msp(tea):
    """Single price ($/t) for all three liquid products that gives NPV = 0 at the IRR."""
    return _solve_scale(tea, [1.0] * len(PRODUCTS)) * 1000


def allocated_msps(tea):
    """Product MSPs ($/t) with costs allocated by market value: every product price is scaled by the same
    factor f (f = 1 means the plant just earns the 10 % IRR at today's prices), so MSP_i = f x market price_i.
    Returns ({product: MSP}, f)."""
    f = _solve_scale(tea, [prices[p] for p in PRODUCTS])
    return {p: f * prices[p] * 1000 for p in PRODUCTS}, f


def product_msps(tea):
    out = {}
    for p in PRODUCTS:
        s = S(tea.system, p); old = s.price
        try:
            out[p] = tea.solve_price(s) * 1000
        except Exception:
            out[p] = np.nan
        s.price = old
    return out


def cost_breakdown(tea, msp):
    """Annual costs per tonne of liquid products at the mean MSP; components add up to the MSP."""
    sysm = tea.system; Q = product_flow_tpy(tea); h = tea.operating_days * 24
    btg = U(sysm, "Turbogenerator")
    rows = {"PS feedstock": S(sysm, "PS_Feed").cost * h,
            "ZnO catalyst": S(sysm, "catalyst").cost * h,
            "Fluidising N2": S(sysm, "fluid_gas").cost * h,
            "Natural gas (reactor heater)": S(sysm, "pyrolyzer_ng").cost * h,
            "Natural gas (boiler)": btg.ins[3].F_mass * prices["NG"] * h,
            "Boiler chemicals, lime, ash": sum(s.cost for s in (btg.ins[4], btg.ins[5])) * h,
            "Utilities (net electricity, cooling)": tea.utility_cost,
            "Fixed O&M incl. labour": tea.FOC}
    rows = {k: v / Q for k, v in rows.items()}
    rows["Capital charge (depreciation, tax, return)"] = msp - sum(rows.values())
    return rows

#%% LCA
ef = pd.read_csv(find_file("impact_factors.csv"), index_col=0)
CATS = list(ef.columns[:-3])
GWP = "Climate change (kg CO2 eq)"


def product_factor(ID, system):
    """Impact factor (per kg) of the conventional product each recovered stream displaces."""
    if ID == "Styrene":
        return ef.loc["PSO", CATS]                        # 'PSO' = styrene (C8H8) row in impact_factors.csv
    if ID == "Aromatics":
        return ef.loc["AromaticsO", CATS]
    if ID == "BTX":                                       # weighted by the BTX stream composition
        s = S(system, "BTX"); m = {"BenzeneO": s.imass["Benzene"], "TolueneO": s.imass["Toluene"],
                                   "XyleneO": s.imass["Ethylbenzene"]}          # xylene = C8 proxy for EB
        tot = sum(m.values()) or 1
        return sum(ef.loc[k, CATS] * v / tot for k, v in m.items())
    raise KeyError(ID)


def get_impacts(system, credits=True):
    """Impacts per hour by contribution. NG and electricity factors are per MJ."""
    btg = U(system, "Turbogenerator")
    ng_MJ = {"Natural gas (reactor heater)": S(system, "pyrolyzer_ng").F_mass * A["NG_LHV"],
             "Natural gas (boiler)": btg.ins[3].F_mass * A["NG_LHV"]}
    net_kW = sum(u.power_utility.rate for u in system.units)
    # natural gas: supply chain (factor per MJ) and its own combustion CO2 shown separately, so the gas burden is
    # not hidden inside the stack emissions
    imp = {"Natural gas supply (upstream)": ef.loc["boiler_turbogenerator_ng", CATS] * sum(ng_MJ.values())}
    imp["Electricity (net)"] = ef.loc["Electricity", CATS] * net_kW * 3.6
    emis = [btg.outs[0], S(system, "pyrolyzer_flue"), S(system, "regenerator_flue")]
    co2e = sum(s.imass["CO2"] + A["GWP_CH4"] * s.imass["CH4"] for s in emis)
    ng_kg = S(system, "pyrolyzer_ng").F_mass + btg.ins[3].F_mass
    co2_ng = min(ng_kg * 44.01 / 16.04, co2e)                 # CH4 + 2 O2 -> CO2 + 2 H2O, complete combustion
    for name, v in [("CO2 from burning natural gas", co2_ng),
                    ("CO2 from burning off-gas, heavies, char", co2e - co2_ng)]:
        d = pd.Series(0., index=CATS); d[GWP] = v; d["Climate change - Fossil (kg CO2 eq)"] = v
        imp[name] = d
    if credits:
        imp["Avoided PS waste treatment"] = -ef.loc["PS_Feed", CATS] * S(system, "PS_Feed").F_mass
        for p in PRODUCTS:
            imp[f"Displaced {p}"] = -product_factor(p, system) * S(system, p).F_mass
    return pd.DataFrame(imp)


def gwp_results(system):
    F = S(system, "PS_Feed").F_mass; P = sum(S(system, p).F_mass for p in PRODUCTS)
    net = get_impacts(system, credits=True).loc[GWP].sum()
    process = get_impacts(system, credits=False).loc[GWP].sum()
    return {"Net GWP (kg CO2e/kg PS)": net / F,
            "Process GWP (kg CO2e/kg PS)": process / F,
            "Process GWP (kg CO2e/kg liquid product)": process / P}

#%% Baseline simulation
def run(system):
    system.simulate()
    return system

t0 = time.time()
ps_sys = run(create_ps_sys())
print(f"Baseline simulated in {time.time() - t0:.1f} s")
pyro = U(ps_sys, "Pyrolyzer")
feedF = S(ps_sys, "PS_Feed").F_mass


def mass_balance(system):
    fin = sum(s.F_mass for s in system.feeds); fout = sum(s.F_mass for s in system.products)
    return fin, fout, (fout - fin) / fin * 100

fin, fout, err = mass_balance(ps_sys)
print(f"Mass balance: in {fin:,.0f} kg/h, out {fout:,.0f} kg/h, closure error {err:.3f} %")
print("GP yields used (wt% of PS):", {k: round(v, 1) for k, v in pyro.last_yields.items()})
print(f"Reactor duty {pyro.duty / 1e3 / feedF:.2f} MJ/kg PS, char heat credit "
      f"{pyro.heat_credit / 1e3 / feedF:.2f} MJ/kg PS, NG {S(ps_sys, 'pyrolyzer_ng').F_mass / feedF:.3f} kg/kg PS")
for p in PRODUCTS:
    print(f"  {p:10s} {S(ps_sys, p).F_mass:8,.0f} kg/h  ({S(ps_sys, p).F_mass / feedF * 100:.1f} % of feed)")
try:
    ps_sys.diagram(format="png", file=os.path.join(OUT, "flowsheet"))
except Exception as e:
    print("Diagram skipped (graphviz not available):", type(e).__name__)

#%% Stream table and product distribution figure
key_streams = ["PS_Feed", "PyrolyzerO", "oil_gas", "offgas", "pyrolysis_oil", "BTX", "Styrene", "Aromatics", "Bottoms"]
IDs = [c.ID for c in chemicals]
stream_table = pd.DataFrame({s: [S(ps_sys, s).imass[c] for c in IDs] for s in key_streams}, index=IDs)
stream_table.loc["Total (kg/h)"] = stream_table.sum()
stream_table.loc["T (K)"] = [S(ps_sys, s).T for s in key_streams]
print(stream_table.round(1).loc[(stream_table.round(1) != 0).any(axis=1)])

groups = {"Styrene": ["Styrene"], "BTX": ["Benzene", "Toluene", "Ethylbenzene"], "Other aromatics": ["AMS"],
          "Heavies": ["Heavies"], "Gas": ["CH4", "C2H4", "C2H6", "C3H6", "H2"], "Char": ["Char"]}
show = ["PyrolyzerO", "pyrolysis_oil", "BTX", "Styrene", "Aromatics", "Bottoms", "offgas"]
comp = pd.DataFrame({s: {g: sum(S(ps_sys, s).imass[c] for c in cs) / feedF for g, cs in groups.items()} for s in show})
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bottom = np.zeros(len(show))
for g, c in zip(comp.index, [COL["blue"], COL["orange"], COL["green"], COL["gold"], COL["grey"], "#333333"]):
    ax.bar(show, comp.loc[g], bottom=bottom, color=c, label=g, width=0.65); bottom += comp.loc[g].values
ax.set_ylabel("kg per kg PS feed"); ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_title(f"Key streams at {A['T_C']} °C (GP yields, ZnO)")
plt.xticks(rotation=30, ha="right"); fig.savefig(os.path.join(OUT, "fig_sim_01_stream_composition.png"))

#%% Sankey diagrams: mass and energy balances of the baseline plant
# Saved as interactive HTML (always) and PNG (if the 'kaleido' package is installed: pip install kaleido).
# level="section" groups the units into process sections; level="unit" shows every unit operation.
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    print("plotly not installed (pip install plotly) - Sankey diagrams skipped")

SECTIONS = {
    "Feed_mixer": "Feed preparation", "Handling": "Feed preparation", "Grinder": "Feed preparation",
    "CHScreen": "Feed preparation",
    "Mixer_pyrolyzer": "Pyrolysis reactor", "Pyrolyzer": "Pyrolysis reactor", "Cyclone": "Pyrolysis reactor",
    "Regenerator": "Pyrolysis reactor",     # catalyst loop kept inside one node (no recycle loop drawn)
    "Cooler_CW": "Vapour condensation", "Cooler_chilled": "Vapour condensation",
    "Condenser": "Vapour condensation", "Pump1": "Vapour condensation",
    "DBTX": "Distillation", "DStyrene": "Distillation", "DAromatics": "Distillation",
    "Turbogenerator": "Boiler-turbogenerator"}
STREAM_LABELS = {
    "PS_Feed": "Waste PS", "catalyst": "ZnO make-up", "fluid_gas": "N2 fluidising gas",
    "pyrolyzer_ng": "Natural gas (reactor heater)", "boiler_turbogenerator_ng": "Natural gas (boiler)",
    "pyrolyzer_air": "Air", "regenerator_air": "Air", "boiler_air": "Air",
    "boiler_makeup_water": "Boiler make-up water", "boiler_lime": "Lime", "boiler_chemicals": "Boiler chemicals",
    "Styrene": "Styrene", "BTX": "BTX", "Aromatics": "Other aromatics",
    "pyrolyzer_flue": "Flue gas (reactor heater)", "regenerator_flue": "Flue gas (regenerator)",
    "emissions": "Flue gas (boiler)", "purged_catalyst": "Spent ZnO purge",
    "rejected_water_and_blowdown": "Boiler blowdown", "ash_disposal": "Ash"}
PRODUCT_NODES = {"Styrene", "BTX", "Other aromatics", "Electricity export"}
UTILITY_NODES = {"Steam to process", "Power to process"}
NODE_COLORS = {"input": "#9a9a9a", "process": COL["blue"], "product": COL["green"],
               "emission": COL["orange"], "loss": "#c0392b", "utility": COL["gold"]}


def _node_kind(name, sections):
    if name in sections: return "process"
    if name in PRODUCT_NODES: return "product"
    if name in UTILITY_NODES: return "utility"
    if name.startswith("Losses"): return "loss"
    if name.startswith(("Flue", "Spent", "Ash", "Boiler blowdown")): return "emission"
    return "input"


RECYCLES = ("screen_recycle", "regenerated_catalyst")     # not drawn (they would loop back across the diagram)


def sankey_links(system, basis="mass", level="section", exclude=("N2", "O2", "H2O")):
    """Links (source, target, value) between process sections (or units) and plant inputs/outputs.
    basis="mass"  : t/d, excluding the chemicals in `exclude` (air, fluidising N2 and water would dwarf the
                    process flows; flue gases therefore show only CO2 and unburnt species).
    basis="energy": MW of lower heating value carried by streams, plus electricity and steam/heat supplied by
                    the boiler-turbogenerator. Steam and electricity used by the process are shown as outputs of
                    the boiler (they end up as heat rejected in condensers/coolers), so no loops are drawn;
                    whatever chemical energy does not leave in a stream, product or electricity is shown as losses
                    (stack, flue gas, heat absorbed by depolymerisation and rejected in the vapour coolers).
    The screen recycle and the regenerated-catalyst recycle are not drawn."""
    group = (lambda u: SECTIONS.get(u.ID, u.ID)) if level == "section" else (lambda u: u.ID)
    sections = {group(u) for u in system.units}
    links = {}
    recyc = {}                  # recycle flows: kept out of the drawing but counted in the loss closure

    def add(a, b, v):
        if v > 1e-9 and a != b:
            links[(a, b)] = links.get((a, b), 0) + v

    for s in system.streams:
        if basis == "mass":
            v = (s.F_mass - sum(s.imass[c] for c in exclude if c in chemicals)) * 24 / 1000      # t/d
        else:
            v = s.LHV / 3.6e6                                                                       # MW
        src = group(s.source) if s.source else STREAM_LABELS.get(s.ID, s.ID)
        dst = group(s.sink) if s.sink else STREAM_LABELS.get(s.ID, s.ID)
        if s.source is None and s.sink is None: continue
        if s.ID in RECYCLES:
            if src != dst: recyc[(src, dst)] = recyc.get((src, dst), 0) + v
            continue
        add(src, dst, v)
    if basis == "energy":
        btg = [u for u in system.units if isinstance(u, bst.BoilerTurbogenerator)][0]
        B = group(btg)
        others = [u for u in system.units if u is not btg]
        add(B, "Power to process", sum(max(u.power_utility.rate, 0) for u in others) / 1000)
        add(B, "Steam to process",
            sum(h.duty for u in others for h in u.heat_utilities if h.duty > 0) / 3.6e6)
        net = -sum(u.power_utility.rate for u in system.units) / 1000
        if net > 0: add(B, "Electricity export", net)
        else: add("Grid electricity", B, -net)
        # close each node's balance with a loss link
        for n in sections:
            allf = {**links}
            for k_, v_ in recyc.items(): allf[k_] = allf.get(k_, 0) + v_
            gap = sum(v for (a, b), v in allf.items() if b == n) - sum(v for (a, b), v in allf.items() if a == n)
            if gap > 1e-6: add(n, "Losses (heat rejected, stack)", gap)
    return links, sections


def plot_sankey(system, basis="mass", level="section", fname=None, title=""):
    links, sections = sankey_links(system, basis, level)
    unit = "t/d" if basis == "mass" else "MW"
    names = list(dict.fromkeys([a for a, b in links] + [b for a, b in links]))
    tot_in = {n: sum(v for (a, b), v in links.items() if b == n) for n in names}
    tot_out = {n: sum(v for (a, b), v in links.items() if a == n) for n in names}
    kinds = [_node_kind(n, sections) for n in names]
    # short names on two lines (name / value) keep the labels inside the gaps between the columns at a font that
    # stays readable when the figure is printed 16 cm wide
    SHORT = {"Feed preparation": "Feed prep.", "Pyrolysis reactor": "Pyrolysis", "Vapour condensation": "Condensation",
             "Boiler-turbogenerator": "Boiler/TG", "Natural gas (reactor heater)": "Natural gas (heater)",
             "Flue gas (reactor heater)": "Flue gas (heater)", "Losses (heat rejected, stack)": "Losses"}
    labels = [f"{SHORT.get(n, n)}<br>{max(tot_in[n], tot_out[n]):,.1f} {unit}" for n in names]

    def rgba(h, a):
        h = h.lstrip("#"); return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
    link_col = [rgba(NODE_COLORS[_node_kind(b, sections)] if _node_kind(b, sections) != "process"
                     else NODE_COLORS[_node_kind(a, sections)] if _node_kind(a, sections) != "process"
                     else COL["blue"], 0.35) for a, b in links]
    rows = pd.DataFrame([{"From": a, "To": b, f"Flow ({unit})": v} for (a, b), v in links.items()])
    if go is None:
        return rows
    W_, H_ = (1300, 1050) if level == "unit" else (1100, 820)
    M_ = dict(l=20, r=20, t=50, b=20)
    pos, pad_px = {}, 26
    if level == "section":
        # Fixed node positions: the boiler-turbogenerator column sits well left of the outputs so its label does
        # not run into theirs, and small streams get enough vertical room for their two-line labels.
        COLX = {"Feed preparation": 0.15, "Pyrolysis reactor": 0.30, "Vapour condensation": 0.45,
                "Distillation": 0.585, "Boiler-turbogenerator": 0.71}
        ORDER = ["Waste PS", "ZnO make-up", "Natural gas (reactor heater)", "Boiler chemicals", "Lime",
                 "Natural gas (boiler)", "Grid electricity",
                 "Flue gas (reactor heater)", "Flue gas (regenerator)", "BTX", "Styrene", "Other aromatics",
                 "Power to process", "Steam to process", "Electricity export", "Flue gas (boiler)",
                 "Losses (heat rejected, stack)", "Boiler blowdown", "Spent ZnO purge", "Ash"]
        Hpx = H_ - M_["t"] - M_["b"]
        val = {n: max(tot_in[n], tot_out[n]) for n in names}
        cols = {}
        for n, k in zip(names, kinds):
            cols.setdefault(COLX.get(n, 0.001 if k == "input" else 0.999), []).append(n)
        for c in cols.values():
            c.sort(key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
        LBL = 52                                    # px needed between the centres of two small nodes
        def stack(c, kpx):                          # top-down stacking, returns centres (px) and bottom (px)
            y, out = 0.0, {}
            for i, n in enumerate(c):
                h = val[n] * kpx
                if i: y = max(y + pad_px, prev_c + LBL - h / 2) if h < LBL else y + pad_px
                out[n] = y + h / 2; prev_c = out[n]; y += h
            return out, y
        kpx = min((Hpx - pad_px * (len(c) - 1)) / sum(val[n] for n in c) for c in cols.values())
        while max(stack(c, kpx)[1] for c in cols.values()) > Hpx: kpx *= 0.97
        for x, c in cols.items():
            cen, _ = stack(c, kpx)
            for n in c: pos[n] = (x, cen[n] / Hpx)
        # plotly rescales node heights with its own padding: match its scale to ours
        lim = max(cols.values(), key=lambda c: stack(c, kpx)[1])
        pad_px = max(4, (Hpx - kpx * sum(val[n] for n in lim)) / max(len(lim) - 1, 1)) * 0.98
        # condensation and distillation sit below the flue-gas links that leave the top of the reactor node
        off = sum(v for (a, b), v in links.items() if a == "Pyrolysis reactor" and b.startswith("Flue gas"))
        for n in ("Vapour condensation", "Distillation"):
            if n in pos and off: pos[n] = (pos[n][0], pos[n][1] + (off * kpx + 1.5 * pad_px) / Hpx)
        # the boiler goes beside the outputs it feeds
        if "Boiler-turbogenerator" in pos:
            outs = [(b, v) for (a, b), v in links.items() if a == "Boiler-turbogenerator" and b in pos]
            yb = sum(pos[b][1] * v for b, v in outs) / sum(v for b, v in outs)
            pos["Boiler-turbogenerator"] = (COLX["Boiler-turbogenerator"], yb)
    fig = go.Figure(go.Sankey(
        arrangement="fixed" if pos else "snap", valueformat=",.1f", valuesuffix=f" {unit}",
        node=dict(pad=pad_px, thickness=16, line=dict(color="black", width=0.4), label=labels,
                  color=[NODE_COLORS[k] for k in kinds],
                  **({"x": [pos[n][0] for n in names], "y": [pos[n][1] for n in names]} if pos else {})),
        link=dict(source=[names.index(a) for a, b in links], target=[names.index(b) for a, b in links],
                  value=list(links.values()), color=link_col)))
    fig.update_layout(title_text=title, font=dict(size=19, family="Arial"), width=W_, height=H_, margin=M_,
                      paper_bgcolor="white")
    if fname:
        fig.write_html(os.path.join(OUT, fname + ".html"), include_plotlyjs=True)
        try:
            fig.write_image(os.path.join(OUT, fname + ".png"), scale=2)
        except Exception as e:
            print(f"{fname}.png not written ({type(e).__name__}); open the .html, or pip install kaleido")
    return rows


sankey_tables = {}
for basis in ("mass", "energy"):
    for level in ("section", "unit"):
        t_ = (f"{'Mass (t/d, excluding air, N2 and water)' if basis == 'mass' else 'Energy (MW, lower heating value)'}"
              f" - {A['capacity_tpd']} t/d PS, {A['T_C']} °C, ZnO")
        sankey_tables[f"Sankey {basis} {level}"] = plot_sankey(
            ps_sys, basis, level, f"fig_sim_00{'a' if basis == 'mass' else 'b'}_sankey_{basis}_{level}", t_)
ls_, _ = sankey_links(ps_sys, "energy")
e_in = sum(v for (a, b), v in ls_.items() if a in ("Waste PS", "Natural gas (reactor heater)", "Natural gas (boiler)"))
print(f"Energy Sankey: inputs {e_in:,.1f} MW LHV; products "
      f"{sum(v for (a, b), v in ls_.items() if b in ('Styrene', 'BTX', 'Other aromatics')):,.1f} MW; "
      f"electricity export {ls_.get(('Boiler-turbogenerator', 'Electricity export'), 0):,.1f} MW")

#%% Process flow diagram (PFD) with stream numbers and stream table
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch


def draw_pfd(system, fname="fig_sim_00_pfd"):
    """Draws the process flow diagram of the simulated plant (matplotlib only, no graphviz) and a stream table
    with the simulated flows. Saved as PNG, PDF and SVG; the stream table also goes to the Excel file."""
    st = lambda ID: S(system, ID)
    un = lambda ID: U(system, ID)
    # Drawn for a landscape page (~25 cm wide): equipment carries short tags (M-101, R-101 ...) explained in the
    # equipment list under the drawing, so every label can be printed at ~8 pt without crowding.
    FS, FS_TAG, FS_NUM = 11.5, 10, 10
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 28.6); ax.set_ylim(-3.7, 10.9); ax.axis("off"); ax.set_aspect("equal")
    LW, EC, FC = 1.4, "#1f2d3d", "#eef3fa"
    for x0, x1, name in [(0.1, 6.5, "Feed preparation"), (6.5, 11.6, "Pyrolysis & regen."),
                         (11.6, 16.6, "Vapour recovery"), (16.6, 22.7, "Product separation"),
                         (22.7, 25.3, "Utilities")]:
        ax.add_patch(Rectangle((x0, 0.3), x1 - x0, 10.45, fc="#f7f7f7", ec="#d0d0d0", lw=0.8, zorder=0))
        ax.text((x0 + x1) / 2, 10.45, name, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#444")

    def tag(x, y, t, va="top", ha="center", rot=0, fs=FS_TAG, inside=False):
        ax.text(x, y, t, ha=ha, va=va, fontsize=fs, fontweight="bold", color="#1f2d3d", rotation=rot, zorder=8,
                bbox=dict(fc="white", ec="none", pad=0.6) if inside else None)

    def mixer(x, y, t):
        ax.add_patch(Circle((x, y), 0.45, fc=FC, ec=EC, lw=LW, zorder=3))
        tag(x, y, t, va="center", fs=8.5)

    def box(x, y, w, h, t="", rounded=False, rot=0):
        p_ = (FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=FC, ec=EC,
                             lw=LW, zorder=3) if rounded else Rectangle((x - w / 2, y - h / 2), w, h, fc=FC, ec=EC, lw=LW, zorder=3))
        ax.add_patch(p_)
        if t: tag(x, y, t, va="center", rot=rot)

    def hx(x, y, t):
        ax.add_patch(Circle((x, y), 0.32, fc=FC, ec=EC, lw=LW, zorder=3))
        ax.plot([x - 0.32, x - 0.1, x + 0.1, x + 0.32], [y - 0.05, y + 0.18, y - 0.18, y + 0.05], color=EC, lw=1, zorder=4)
        tag(x, y - 0.42, t)

    def column(x, y, h, t):
        box(x, y, 0.6, h, rounded=True)
        for k in range(1, 9):
            yy = y - h / 2 + k * h / 9
            ax.plot([x - 0.3, x + 0.3], [yy, yy], color="#9aa7b5", lw=0.6, zorder=4)
        tag(x, y, t, va="center", rot=90, inside=True)

    def arrow(pts, num=None, at=0, text=None, text_xy=None, ha="left", dashed=False, color="#2a4d77", va="center", fs=None):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=1.4, ls="--" if dashed else "-", zorder=2, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=13), zorder=5)
        if num is not None:
            (x0, y0), (x1, y1) = pts[at], pts[at + 1]
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            d_ = 0.28
            ax.add_patch(Polygon([(xm, ym + d_), (xm + d_, ym), (xm, ym - d_), (xm - d_, ym)], fc="white",
                                 ec=color, lw=1, zorder=6))
            ax.text(xm, ym, str(num), ha="center", va="center", fontsize=FS_NUM if num < 10 else FS_NUM - 1.5,
                    color=color, zorder=7, fontweight="bold")
        if text:
            tx, ty = text_xy if text_xy else pts[-1]
            ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs or FS, zorder=6, linespacing=1.0)

    F = lambda ID: st(ID).F_mass
    kgh = lambda ID: f"{F(ID):,.0f} kg/h"
    pyro, btg = un("Pyrolyzer"), un("Turbogenerator")
    stages = lambda ID: un(ID).design_results.get("Actual stages", np.nan)
    net_MW = -sum(u.power_utility.rate for u in system.units) / 1000

    # ---- equipment ----
    mixer(1.55, 5, "M-101")
    ax.add_patch(Polygon([(2.55, 5.4), (3.25, 5.4), (3.02, 4.65), (2.78, 4.65)], fc=FC, ec=EC, lw=LW, zorder=3))
    tag(2.9, 4.55, "C-101")
    box(4.25, 5, 0.8, 0.75)
    for dx in (-0.16, 0.16):
        ax.add_patch(Circle((4.25 + dx, 5), 0.14, fc="white", ec=EC, lw=1, zorder=4))
    tag(4.25, 4.52, "G-101")
    box(5.6, 5, 0.8, 0.65)
    ax.plot([5.2, 6.0], [5.18, 4.82], color=EC, lw=1, ls=(0, (2, 1.5)), zorder=4)
    tag(5.6, 4.57, "S-101")
    mixer(7.3, 5, "M-102")
    box(8.9, 5, 0.9, 2.2, "R-101", rot=90)
    cx, cy = 10.6, 5.6
    ax.add_patch(Polygon([(cx - 0.35, cy + 0.6), (cx + 0.35, cy + 0.6), (cx + 0.35, cy), (cx + 0.08, cy - 0.8),
                          (cx - 0.08, cy - 0.8), (cx - 0.35, cy)], fc=FC, ec=EC, lw=LW, zorder=3))
    tag(cx + 0.42, cy - 0.45, "CY-101", ha="left", va="center")
    box(10.6, 2.3, 1.0, 1.2, "R-102", rot=90)
    hx(12.2, 6.6, "E-101")
    hx(13.5, 6.6, "E-102")
    box(14.8, 6.2, 0.7, 1.6, "V-101", rounded=True, rot=90)
    ax.add_patch(Circle((15.9, 4.0), 0.3, fc=FC, ec=EC, lw=LW, zorder=3))
    ax.add_patch(Polygon([(15.75, 3.8), (15.75, 4.2), (16.15, 4.0)], fc=EC, zorder=4))
    tag(15.9, 3.6, "P-101")
    column(17.5, 5.4, 4.2, "T-101")
    column(19.6, 4.8, 3.6, "T-102")
    column(21.7, 4.4, 2.8, "T-103")
    box(23.8, 8.7, 1.8, 1.1, "B-101")

    # ---- streams ----
    n = 0
    table = []

    def stream(ID, pts, desc, **kw):
        nonlocal n
        n += 1
        s = st(ID)
        comp = (pd.Series({c.ID: s.imass[c.ID] for c in chemicals}) / max(s.F_mass, 1e-12) * 100).sort_values(ascending=False)
        comp = comp[comp > 0.5].head(4)
        table.append({"Stream": n, "Description": desc, "ID in model": ID, "T (°C)": s.T - 273.15, "P (bar)": s.P / 1e5,
                      "Mass flow (kg/h)": s.F_mass,
                      "Main components (wt%)": ", ".join(f"{k} {v:.0f}" for k, v in comp.items())})
        arrow(pts, num=n, **kw)

    stream("PS_Feed", [(0.2, 5), (1.1, 5)], "Waste PS feed", text=f"Waste PS\n{A['capacity_tpd']} t/d",
           text_xy=(0.2, 5.95), ha="left")
    stream(un("Feed_mixer").outs[0].ID, [(2.0, 5), (2.62, 5)], "Mixed feed")
    stream(un("Handling").outs[0].ID, [(3.17, 5), (3.85, 5)], "Conveyed PS")
    stream(un("Grinder").outs[0].ID, [(4.65, 5), (5.2, 5)], "Ground PS")
    stream("screen_recycle", [(5.6, 4.67), (5.6, 3.7), (1.55, 3.7), (1.55, 4.55)], "Screen oversize (recycle)", at=1)
    stream("CRPS", [(6.0, 5), (6.85, 5)], "Screened PS")
    stream("catalyst", [(7.1, 7.2), (7.1, 5.4)], "ZnO make-up", text="ZnO", text_xy=(7.1, 7.45), ha="center")
    stream("fluid_gas", [(7.5, 7.95), (7.5, 5.4)], "N2 fluidising gas (make-up)", at=0,
           text="N$_2$", text_xy=(7.5, 8.2), ha="center")
    stream(un("Mixer_pyrolyzer").outs[0].ID, [(7.75, 5), (8.45, 5)], "Reactor feed (PS + ZnO + N2)")
    stream("pyrolyzer_ng", [(8.75, 2.65), (8.75, 3.9)], "Natural gas to reactor heater", at=0)
    arrow([(9.05, 2.65), (9.05, 3.9)])
    ax.text(8.9, 2.4, "NG + air", ha="center", va="center", fontsize=FS)
    stream("pyrolyzer_flue", [(8.9, 6.1), (8.9, 7.2)], "Flue gas, reactor heater",
           text="Flue gas", text_xy=(8.9, 7.45), ha="center")
    stream("PyrolyzerO", [(9.35, 5.9), (10.25, 5.9)], "Pyrolysis vapours + solids")
    stream("spent_solids", [(10.6, 4.8), (10.6, 2.9)], "Spent catalyst + char", at=0)
    stream("regenerated_catalyst", [(10.1, 1.95), (7.3, 1.95), (7.3, 4.55)], "Regenerated ZnO (recycle)", at=0)
    stream("regenerator_air", [(12.0, 1.9), (11.1, 1.9)], "Air to regenerator", text="Air", text_xy=(12.1, 1.9))
    stream("regenerator_flue", [(11.1, 2.7), (12.0, 2.7)], "Flue gas, regenerator", text="Flue gas", text_xy=(12.1, 2.7))
    stream("purged_catalyst", [(10.6, 1.7), (10.6, 0.8)], "Spent ZnO purge", text="ZnO purge",
           text_xy=(10.8, 0.85), ha="left")
    stream("oil_gas", [(10.6, 6.2), (10.6, 6.6), (11.88, 6.6)], "Pyrolysis vapours", at=1)
    stream(un("Cooler_CW").outs[0].ID, [(12.52, 6.6), (13.18, 6.6)], "Partly condensed vapours (47 °C)")
    stream(un("Cooler_chilled").outs[0].ID, [(13.82, 6.6), (14.45, 6.6)], "Gas + condensate (10 °C)")
    stream("offgas", [(14.8, 7.0), (14.8, 8.7), (22.9, 8.7)], "Non-condensable gas to boiler", at=1)
    stream("pyrolysis_oil", [(14.8, 5.4), (14.8, 4.0), (15.6, 4.0)], "Pyrolysis oil", at=0)
    stream(un("Pump1").outs[0].ID, [(16.2, 4.0), (16.7, 4.0), (16.7, 5.0), (17.2, 5.0)], "Oil to BTX column", at=1)
    stream("BTX", [(17.5, 7.5), (17.5, 7.75), (25.3, 7.75)], "BTX product", at=1,
           text=f"BTX\n{kgh('BTX')}", text_xy=(25.45, 7.75), fs=10.5)
    stream(un("DBTX").outs[1].ID, [(17.5, 3.3), (17.5, 2.3), (19.0, 2.3), (19.0, 4.6), (19.3, 4.6)], "BTX-column bottoms", at=1)
    stream("Styrene", [(19.6, 6.6), (19.6, 6.85), (25.3, 6.85)], "Styrene product", at=1,
           text=f"Styrene\n{kgh('Styrene')}", text_xy=(25.45, 6.85), fs=10.5)
    stream(un("DStyrene").outs[1].ID, [(19.6, 3.0), (19.6, 1.9), (21.1, 1.9), (21.1, 4.2), (21.4, 4.2)], "Styrene-column bottoms", at=1)
    stream("Aromatics", [(21.7, 5.8), (21.7, 5.95), (25.3, 5.95)], "Other aromatics product", at=1,
           text=f"Other aromatics\n{kgh('Aromatics')}", text_xy=(25.45, 5.95), fs=10.5)
    stream("Bottoms", [(21.7, 3.0), (21.7, 1.5), (24.45, 1.5), (24.45, 8.15)], "Heavies to boiler", at=1)
    stream(btg.ins[3].ID, [(23.2, 10.0), (23.2, 9.25)], "Natural gas to boiler (if needed)",
           text="NG", text_xy=(22.85, 9.75), ha="right")
    stream(btg.outs[0].ID, [(24.1, 9.25), (24.1, 10.0)], "Boiler flue gas", text="Flue gas", text_xy=(24.45, 9.75))
    arrow([(24.7, 8.7), (25.3, 8.7)], text=f"Electricity\n{net_MW:.1f} MW net", text_xy=(25.45, 8.75), fs=10.5)
    arrow([(22.9, 8.3), (20.9, 8.3)], dashed=True, color="#b07800")
    ax.text(20.8, 8.3, "steam to reboilers ", ha="right", va="center", fontsize=10, color="#b07800")

    # ---- equipment list ----
    T_r = pyro.T - 273.15
    items = [("M-101", "Feed mixer"), ("C-101", "Conveyor and hopper"), ("G-101", "Grinder (50 kWh/t)"),
             ("S-101", "Screen (1% oversize recycled)"), ("M-102", "Reactor feed mixer"),
             ("R-101", f"Fluidised-bed pyrolyzer, {T_r:.0f} °C, ZnO, NG-fired"), ("CY-101", "Cyclone"),
             ("R-102", "Catalyst regenerator (char burn-off)"),
             ("E-101", "Vapour cooler, cooling water, 47 °C"), ("E-102", "Vapour cooler, chilled brine, 10 °C"),
             ("V-101", "Flash drum"), ("P-101", "Oil pump"),
             ("T-101", f"BTX column ({stages('DBTX'):.0f} trays)"), ("T-102", f"Styrene column ({stages('DStyrene'):.0f} trays)"),
             ("T-103", f"Aromatics column ({stages('DAromatics'):.0f} trays)"),
             ("B-101", "Boiler-turbogenerator (burns off-gas and heavies)")]
    for i, (t_, d_) in enumerate(items):
        col, row = divmod(i, 8)
        x0, y0 = 0.3 + col * 14.2, -0.1 - row * 0.45
        ax.text(x0, y0, t_, fontsize=10.5, fontweight="bold", va="center", color="#1f2d3d")
        ax.text(x0 + 1.65, y0, d_, fontsize=10.5, va="center", color="#222")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"))

    # stream table figure
    tdf = pd.DataFrame(table)
    show_ = tdf.drop(columns="ID in model").copy()
    show_["T (°C)"] = show_["T (°C)"].map(lambda v: f"{v:,.0f}")
    show_["P (bar)"] = show_["P (bar)"].map(lambda v: f"{v:.2f}")
    show_["Mass flow (kg/h)"] = show_["Mass flow (kg/h)"].map(lambda v: f"{v:,.0f}")
    fig2, ax2 = plt.subplots(figsize=(7, 0.24 * len(show_) + 0.6)); ax2.axis("off")
    tb = ax2.table(cellText=show_.values, colLabels=show_.columns, loc="center", cellLoc="left",
                   colWidths=[0.05, 0.27, 0.06, 0.06, 0.1, 0.46])
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.15)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0: cell.set_facecolor("#e3ecf7"); cell.set_text_props(fontweight="bold")
    fig2.savefig(os.path.join(OUT, f"{fname}_stream_table.png"))
    return tdf

pfd_table = draw_pfd(ps_sys)
print(pfd_table[["Stream", "Description", "Mass flow (kg/h)"]].round(0).to_string(index=False))

#%% TEA: minimum selling prices and cost breakdown
tea = get_tea(ps_sys)
msp0 = mean_msp(tea)
msps0 = product_msps(tea)                       # each product alone, the other two at market price
alloc0, f0 = allocated_msps(tea)                # market-value allocation (same factor for all products)
breakdown = cost_breakdown(tea, msp0)
print(f"Installed equipment cost {tea.DPI / 1e6:.1f} MM$, FCI {tea.FCI / 1e6:.1f} MM$, "
      f"TCI {tea.TCI / 1e6:.1f} MM$, AOC {tea.AOC / 1e6:.1f} MM$/yr")
print(f"Mean liquid-product MSP: ${msp0:,.0f}/t   (market-price-weighted average "
      f"${sum(prices[p] * S(ps_sys, p).F_mass for p in PRODUCTS) / sum(S(ps_sys, p).F_mass for p in PRODUCTS) * 1000:,.0f}/t)")
print("Single-product MSPs (others at market price), $/t:", {k: round(v) for k, v in msps0.items()})
print(f"Market-value-allocated MSPs (price factor {f0:.2f} x market), $/t:", {k: round(v) for k, v in alloc0.items()})
msp_table = pd.DataFrame({
    "Market price ($/t)": {p: prices[p] * 1000 for p in PRODUCTS},
    "MSP, market-value allocation ($/t)": alloc0,
    "MSP, other products at market price ($/t)": msps0,
    "Product flow (t/yr)": {p: S(ps_sys, p).get_total_flow("tonne/year") for p in PRODUCTS}})
msp_table.loc["Mean liquid product"] = [np.nan, msp0, np.nan, product_flow_tpy(tea)]
print(msp_table.round(0))

# Product MSPs (market-value allocation) vs market prices
fig, ax = plt.subplots(figsize=(3.6, 3.0))
xx = np.arange(len(PRODUCTS)); w = 0.38
for j, (vals, c, lab) in enumerate([([prices[p] * 1000 for p in PRODUCTS], COL["grey"], "Market price"),
                                    ([alloc0[p] for p in PRODUCTS], COL["blue"], "MSP (market-value allocation)")]):
    bars_ = ax.bar(xx + (j - 0.5) * w, vals, w, color=c, label=lab)
    for r_ in bars_:
        ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 12, f"{r_.get_height():,.0f}", ha="center",
                va="bottom", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels([{"Styrene": "Styrene", "BTX": "BTX", "Aromatics": "Other\naromatics"}[p] for p in PRODUCTS]); ax.set_ylabel("\\$ per tonne")
ax.set_ylim(0, max(prices[p] for p in PRODUCTS) * 1000 * 1.45)
ax.legend(frameon=False, loc="upper left", ncol=1, bbox_to_anchor=(0, 1.02))
fig.savefig(os.path.join(OUT, "fig_sim_02b_styrene_btx_msp.png"))

fig, ax = plt.subplots(figsize=(6.5, 2.5))
b = pd.Series(breakdown); pos, neg = 0, 0
pal = [COL["blue"], COL["green"], "#9fd8c2", COL["orange"], "#f3a27a", "#c9c9c9", COL["gold"], "#6aa6e8", "#555555"]
for (k, v), c in zip(b.items(), pal):
    if v >= 0: ax.barh(0, v, left=pos, color=c, label=k, height=0.5); pos += v
    else: ax.barh(0, v, left=neg, color=c, label=k, height=0.5); neg += v
ax.axvline(msp0, color="k", lw=1); ax.text(msp0, 0.32, f"  mean MSP \\${msp0:,.0f}/t", va="bottom", ha="right", fontsize=9)
ax.set_yticks([]); ax.set_xlabel("\\$ per tonne of liquid products"); ax.set_ylim(-0.4, 0.55)
ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.32), loc="upper center", ncol=3, fontsize=8)
fig.savefig(os.path.join(OUT, "fig_sim_02_msp_breakdown.png"))

#%% TEA: feedstock price sensitivity
feed_prices = [0.0, 0.23, prices["PS"], 0.59, 0.64, 0.90]   # 0.90 = virgin-PS index (old base case)
fp_rows = []
for fp in feed_prices:
    S(ps_sys, "PS_Feed").price = fp
    al, _ = allocated_msps(tea)
    fp_rows.append({"PS price ($/t)": fp * 1000, "Mean MSP ($/t)": mean_msp(tea),
                    **{f"{k} MSP, allocated ($/t)": al[k] for k in MSP_PRODUCTS},
                    **{f"{k} MSP, others at market ($/t)": v for k, v in product_msps(tea).items()}})
S(ps_sys, "PS_Feed").price = prices["PS"]
fp_df = pd.DataFrame(fp_rows); print(fp_df.round(0).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(7, 3.0))
for ax, method, title in [(axs[0], "allocated", "Market-value allocation"),
                          (axs[1], "others at market", "One product solved, others at market")]:
    for p, c, dy in zip(MSP_PRODUCTS, [COL["blue"], COL["orange"]], [6, -6]):
        ax.plot(fp_df["PS price ($/t)"], fp_df[f"{p} MSP, {method} ($/t)"], "o-", color=c, label=f"{p} MSP")
        ax.axhline(prices[p] * 1000, ls="--", color=c, lw=0.8, alpha=0.7)
        ax.annotate(f"{p} market \\${prices[p] * 1000:,.0f}/t", (fp_df["PS price ($/t)"].max() * 0.55, prices[p] * 1000), xytext=(0, dy),
                    textcoords="offset points", color=c, fontsize=8, va="bottom" if dy > 0 else "top")
    ax.axhline(0, color="k", lw=0.6); ax.set_title(title, fontsize=10)
    ax.set_xlabel("Waste PS price (\\$/t)"); ax.set_ylabel("Minimum selling price (\\$/t)")
axs[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sim_03_feed_price.png"))

#%% TEA: PS price sensitivity of the styrene, BTX and other-aromatics MSPs
# NPV is linear in the feed price, so the flowsheet is not re-simulated; only the PS_Feed price changes.
LABEL = {"Styrene": "Styrene", "BTX": "BTX", "Aromatics": "Other aromatics"}
PCOL = {"Styrene": COL["blue"], "BTX": COL["orange"], "Aromatics": COL["green"]}
ps_grid = np.linspace(-0.2, 1.2, 15)                       # $/kg waste PS; < 0 = gate fee paid to the plant
ps_rows = []
for fp in ps_grid:
    S(ps_sys, "PS_Feed").price = fp
    al, fac = allocated_msps(tea); om = product_msps(tea)
    ps_rows.append({"PS price ($/t)": fp * 1000, "Price factor (MSP / market)": fac,
                    **{f"{LABEL[p]} MSP, allocated ($/t)": al[p] for p in PRODUCTS},
                    **{f"{LABEL[p]} MSP, others at market ($/t)": om[p] for p in PRODUCTS}})
S(ps_sys, "PS_Feed").price = prices["PS"]
ps_df = pd.DataFrame(ps_rows)


def breakeven_ps(tea):
    """Highest waste PS price ($/t) at which the plant still earns the IRR with all products at market price."""
    s = S(tea.system, "PS_Feed"); old = s.price
    try:
        return tea.solve_price(s) * 1000
    finally:
        s.price = old

be_ps = breakeven_ps(tea)
print(f"Break-even waste PS price (all products at market price, IRR 10 %): ${be_ps:,.0f}/t")
print(ps_df.round(0).to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(7, 6.0)); axs = axs.ravel()
x = ps_df["PS price ($/t)"]
# (a) market-value allocation: all three products on one axis
ax = axs[0]
for p, mk in zip(PRODUCTS, ["s", "o", "^"]):
    yv = ps_df[f"{LABEL[p]} MSP, allocated ($/t)"]
    ax.plot(x, yv, marker=mk, ms=5, lw=2, color=PCOL[p], label=LABEL[p])
ax.axvline(be_ps, color="k", lw=0.8, ls=":")
ax.text(be_ps, ax.get_ylim()[1], f" break-even PS\n \\${be_ps:,.0f}/t", va="top", fontsize=8)
ax.axvline(prices["PS"] * 1000, color=COL["grey"], lw=0.8, ls="--")
ax.set_title("(a) Market-value allocation")
ax.set_ylabel("Minimum selling price (\\$/t)"); ax.legend(frameon=False, loc="upper left")
# (b-d) one product solved, the other two at market price - one panel per product (own y-scale)
for ax, p, tag in zip(axs[1:], PRODUCTS, "bcd"):
    yv = ps_df[f"{LABEL[p]} MSP, others at market ($/t)"]
    ax.plot(x, yv, "o-", ms=5, lw=2, color=PCOL[p], label="MSP, others at market price")
    ax.plot(x, ps_df[f"{LABEL[p]} MSP, allocated ($/t)"], "--", lw=1.2, color=PCOL[p], alpha=0.6,
            label="MSP, market-value allocation")
    ax.axhline(prices[p] * 1000, color=COL["grey"], lw=0.8, ls="--")
    ax.text(x.iloc[0], prices[p] * 1000, f" market \\${prices[p] * 1000:,.0f}/t", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(be_ps, color="k", lw=0.8, ls=":")
    ax.set_title(f"({tag}) {LABEL[p]} solved, others at market")
    ax.set_ylabel(f"{LABEL[p]} MSP (\\$/t)")
axs[1].legend(frameon=False, fontsize=8, loc="upper left")
for ax in axs:
    ax.set_xlabel("Waste PS price (\\$/t)")
    ax.axvspan(x.min(), 0, color=COL["grey"], alpha=0.12, lw=0)
axs[0].text(x.min(), axs[0].get_ylim()[0], " gate fee", fontsize=8, va="bottom", color=COL["grey"])
axs[0].text(prices["PS"] * 1000, axs[0].get_ylim()[0], " base case", fontsize=8, va="bottom", color=COL["grey"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sim_03c_ps_price_msp_3products.png"))

#%% TEA: break-even map - NPV over waste PS price x styrene price (BTX and other aromatics at market price)
from matplotlib.colors import TwoSlopeNorm
ps_ax = np.linspace(-200, 1200, 15)                    # $/t
sty_ax = np.linspace(400, 2000, 17)                    # $/t
sty_stream, feed_stream = S(ps_sys, "Styrene"), S(ps_sys, "PS_Feed")
npv_grid = np.zeros((len(sty_ax), len(ps_ax)))
for i, sp in enumerate(sty_ax):
    sty_stream.price = sp / 1000
    for j, fp in enumerate(ps_ax):
        feed_stream.price = fp / 1000
        npv_grid[i, j] = tea.NPV / 1e6
sty_stream.price, feed_stream.price = prices["Styrene"], prices["PS"]
npv_df = pd.DataFrame(npv_grid, index=pd.Index(sty_ax, name="Styrene price ($/t)"),
                      columns=pd.Index(ps_ax, name="PS price ($/t)"))
fig, ax = plt.subplots(figsize=(4.8, 3.6))
lim = np.abs(npv_grid).max()
cf = ax.contourf(ps_ax, sty_ax, npv_grid, levels=21, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim))
cs = ax.contour(ps_ax, sty_ax, npv_grid, levels=[0], colors="k", linewidths=2)
ax.clabel(cs, fmt="NPV = 0", fontsize=9)
fig.colorbar(cf, ax=ax, label="NPV at 10 % IRR (MM\\$); blue = profitable")
ax.plot(prices["PS"] * 1000, prices["Styrene"] * 1000, "o", ms=9, mfc="white", mec="k", mew=1.5)
ax.annotate("base case", (prices["PS"] * 1000, prices["Styrene"] * 1000), xytext=(8, -12),
            textcoords="offset points", fontsize=9)
ax.axhline(prices["Styrene"] * 1000, color="k", lw=0.8, ls="--")
ax.text(ps_ax[0], prices["Styrene"] * 1000, f" styrene market \\${prices['Styrene'] * 1000:,.0f}/t", va="bottom", fontsize=8)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Waste PS price (\\$/t)"); ax.set_ylabel("Styrene selling price (\\$/t)")
ax.set_title("BTX and other aromatics at market price")
fig.savefig(os.path.join(OUT, "fig_sim_03d_breakeven_map_ps_styrene.png"))

# main-text figure: (a) product MSPs vs PS price (market-value allocation), (b) NPV map
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axs[0]
for p, mk in zip(PRODUCTS, ["s", "o", "^"]):
    ax.plot(x, ps_df[f"{LABEL[p]} MSP, allocated ($/t)"], marker=mk, ms=3.5, lw=1.6, color=PCOL[p], label=LABEL[p])
ax.axvspan(x.min(), 0, color=COL["grey"], alpha=0.12, lw=0)
ax.axvline(prices["PS"] * 1000, color=COL["grey"], lw=0.8, ls="--"); ax.axvline(be_ps, color="k", lw=0.8, ls=":")
ax.text(be_ps, ax.get_ylim()[0] + 0.05 * np.ptp(ax.get_ylim()), f" break-even\n \\${be_ps:,.0f}/t", fontsize=8, va="bottom")
ax.set_xlabel("Waste PS price (\\$/t)"); ax.set_ylabel("Minimum selling price (\\$/t)")
ax.set_title("(a) Product MSPs"); ax.legend(frameon=False, loc="upper left")
ax = axs[1]
cf = ax.contourf(ps_ax, sty_ax, npv_grid, levels=21, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim))
cs = ax.contour(ps_ax, sty_ax, npv_grid, levels=[0], colors="k", linewidths=1.5)
ax.clabel(cs, fmt="NPV = 0", fontsize=8)
fig.colorbar(cf, ax=ax, label="NPV (MM\\$)")
ax.plot(prices["PS"] * 1000, prices["Styrene"] * 1000, "o", ms=6, mfc="white", mec="k", mew=1.2)
ax.annotate("base case", (prices["PS"] * 1000, prices["Styrene"] * 1000), xytext=(6, -11), textcoords="offset points", fontsize=8)
ax.set_xlabel("Waste PS price (\\$/t)"); ax.set_ylabel("Styrene price (\\$/t)"); ax.set_title("(b) NPV at 10 % IRR")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sim_03g_ps_price_main.png"))

#%% TEA: heat map - styrene MSP (market-value allocation) over waste PS price x reactor temperature
lo_T, hi_T = GP_DATA_RANGE[(A["regime"], A["catalyst"])]
T_ax = np.linspace(lo_T, hi_T, 5)                      # stay inside the ZnO/continuous GP data range
psT_ax = np.linspace(-200, 1200, 8)                    # $/t
heat = np.zeros((len(T_ax), len(psT_ax))); heat_be = []
for i, T_C in enumerate(T_ax):
    s = run(create_ps_sys(T_C=float(T_C), ID="ps_T")); t = get_tea(s)
    for j, fp in enumerate(psT_ax):
        S(s, "PS_Feed").price = fp / 1000
        heat[i, j] = allocated_msps(t)[0]["Styrene"]
    S(s, "PS_Feed").price = prices["PS"]
    heat_be.append(breakeven_ps(t))
heat_df = pd.DataFrame(heat, index=pd.Index(T_ax, name="T (°C)"), columns=pd.Index(psT_ax, name="PS price ($/t)"))
heat_df["Break-even PS price ($/t)"] = heat_be
fig, ax = plt.subplots(figsize=(7, 3.4))
im = ax.imshow(heat, cmap="Blues", aspect="auto", origin="lower")
fig.colorbar(im, ax=ax, label="Styrene MSP, market-value allocation (\\$/t)")
mid = (heat.min() + heat.max()) / 2
for i in range(len(T_ax)):
    for j in range(len(psT_ax)):
        v = heat[i, j]; ok_ = v <= prices["Styrene"] * 1000
        ax.text(j, i, f"{v:,.0f}" + ("*" if ok_ else ""), ha="center", va="center", fontsize=8,
                color="white" if v > mid else "black", fontweight="bold" if ok_ else "normal")
ax.set_xticks(range(len(psT_ax))); ax.set_xticklabels([f"{v:,.0f}" for v in psT_ax])
ax.set_yticks(range(len(T_ax))); ax.set_yticklabels([f"{v:.0f}" for v in T_ax])
ax.set_xlabel("Waste PS price (\\$/t)"); ax.set_ylabel("Reactor temperature (°C)")
ax.set_title(f"Bold * = at or below the styrene market price (\\${prices['Styrene'] * 1000:,.0f}/t)")
fig.savefig(os.path.join(OUT, "fig_sim_03e_heatmap_ps_price_temperature.png"))

#%% TEA: plant capacity sweep - product MSPs and break-even PS price
# (labour scales with capacity^LABOR_SCALING in get_tea; equipment with each unit's cost exponent)
cap_rows = []
for cap in [50, 100, 250, 500, 750, 1000]:
    s = run(create_ps_sys(capacity_tpd=cap, ID="cap")); t = get_tea(s); al = allocated_msps(t)[0]
    cap_rows.append({"Capacity (t/d)": cap, "FCI (MM$)": t.FCI / 1e6, "Mean MSP ($/t)": mean_msp(t),
                     **{f"{LABEL[p]} MSP, allocated ($/t)": al[p] for p in PRODUCTS},
                     "Break-even PS price ($/t)": breakeven_ps(t)})
cap_df = pd.DataFrame(cap_rows); print(cap_df.round(0).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(7, 3.0))
for p, mk in zip(PRODUCTS, ["s", "o", "^"]):
    yv = cap_df[f"{LABEL[p]} MSP, allocated ($/t)"]
    axs[0].plot(cap_df["Capacity (t/d)"], yv, marker=mk, ms=5, lw=2, color=PCOL[p], label=LABEL[p])
axs[0].set_ylabel("MSP (\\$/t)")
axs[0].set_title(f"(a) Product MSPs (PS at \\${prices['PS'] * 1000:,.0f}/t)")
axs[0].legend(frameon=False)
axs[1].plot(cap_df["Capacity (t/d)"], cap_df["Break-even PS price ($/t)"], "o-", lw=2, color=COL["grey"])
for xv, yv in zip(cap_df["Capacity (t/d)"], cap_df["Break-even PS price ($/t)"]):
    axs[1].annotate(f"\\${yv:,.0f}", (xv, yv), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
axs[1].axhline(0, color="k", lw=0.6)
axs[1].set_ylabel("Break-even PS price (\\$/t)")
axs[1].set_title("(b) Break-even waste PS price")
for ax in axs:
    ax.axvline(A["capacity_tpd"], color=COL["grey"], lw=0.8, ls="--"); ax.set_xlabel("Capacity (t PS/d)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sim_03f_capacity_sweep.png"))

#%% Temperature sweep (deterministic GP mean) - ZnO and non-catalytic
sweep_rows = []
cases = {"ZnO": ("basic", np.linspace(*GP_DATA_RANGE[(A["regime"], "basic")], 5)),   # 430-510 °C
         "Non-catalytic": ("none", np.arange(450, 651, 50))}
for label, (cat, Ts) in cases.items():
    for T_C in Ts:
        s = run(create_ps_sys(T_C=float(T_C), catalyst=cat, ID="sweep"))
        t = get_tea(s); y = U(s, "Pyrolyzer").last_yields; ci = gp.interval(T_C, A["regime"], cat)
        sweep_rows.append({"Case": label, "T (°C)": T_C, **{f"{k} (wt%)": v for k, v in y.items()},
                           **{f"{k} 95% low": ci[k][0] for k in ci}, **{f"{k} 95% high": ci[k][1] for k in ci},
                           "Products (t/yr)": product_flow_tpy(t), "Mean MSP ($/t)": mean_msp(t),
                           **{f"{k} MSP, allocated ($/t)": v for k, v in allocated_msps(t)[0].items() if k in MSP_PRODUCTS},
                           **{f"{k} MSP, others at market ($/t)": v for k, v in product_msps(t).items() if k in MSP_PRODUCTS},
                           "Inside GP data range": GP_DATA_RANGE[(A["regime"], cat)][0] <= T_C <= GP_DATA_RANGE[(A["regime"], cat)][1],
                           **gwp_results(s)})
sweep = pd.DataFrame(sweep_rows); print(sweep[["Case", "T (°C)", "Styrene (wt%)", "Mean MSP ($/t)", "Net GWP (kg CO2e/kg PS)"]].round(2).to_string(index=False))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
for (label, d), c in zip(sweep.groupby("Case", sort=False), [COL["blue"], COL["orange"]]):
    axs[0].plot(d["T (°C)"], d["Styrene (wt%)"], "o-", ms=3.5, color=c, label=label)
    axs[0].fill_between(d["T (°C)"], d["Styrene 95% low"], d["Styrene 95% high"], color=c, alpha=0.15)
    axs[1].plot(d["T (°C)"], d["Styrene MSP, allocated ($/t)"], "o-", ms=3.5, color=c)
    axs[1].plot(d["T (°C)"], d["BTX MSP, allocated ($/t)"], "s--", color=c, ms=3)
    axs[2].plot(d["T (°C)"], d["Net GWP (kg CO2e/kg PS)"], "o-", ms=3.5, color=c)
axs[0].set_ylabel("Styrene yield (wt%)"); axs[1].set_ylabel("MSP (\\$/t)")
axs[1].text(0.97, 0.55, "styrene (solid)\nBTX (dashed)", transform=axs[1].transAxes, va="center", ha="right", fontsize=8)
axs[2].set_ylabel("Net GWP (kg CO$_2$e/kg PS)")
lo, hi = GP_DATA_RANGE[(A["regime"], "basic")]
for ax, tag in zip(axs, "abc"):
    ax.axvspan(lo, hi, color=COL["blue"], alpha=0.06); ax.set_xlabel("Temperature (°C)")
    ax.set_title(f"({tag})", loc="left")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "fig_sim_04_temperature_sweep.png"))

#%% Equipment costs
ps_sys = run(create_ps_sys()); tea = get_tea(ps_sys)
# purchase costs; the TEA applies the Lang factor (5.05) to their sum to get the fixed capital investment
eq = pd.Series({u.ID: u.purchase_cost / 1e6 for u in ps_sys.units}).sort_values()
eq = eq[eq > 0]
major = eq[eq > 0.05]; eq_plot = pd.concat([pd.Series({"Other": eq[eq <= 0.05].sum()}), major])
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.barh(eq_plot.index, eq_plot.values, color=COL["blue"])
ax.set_xlabel(f"Purchase cost (MM\\$); FCI = {tea.FCI / 1e6:.0f} MM\\$")
for i, v in enumerate(eq_plot.values): ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
fig.savefig(os.path.join(OUT, "fig_sim_05_equipment_costs.png"))

#%% LCA: baseline contributions
imp = get_impacts(ps_sys) / feedF                    # per kg PS
gwp0 = gwp_results(ps_sys)
print({k: round(v, 3) for k, v in gwp0.items()})
g = imp.loc[GWP].sort_values()
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.barh(g.index, g.values, color=[COL["green"] if v < 0 else COL["orange"] for v in g.values])
ax.axvline(0, color="k", lw=0.8)
for i, v in enumerate(g.values):                       # values on the bars, so small contributions stay visible
    ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=7.5)
ax.set_xlim(g.min() - 0.45, max(g.max(), 0) + 0.45)
ax.set_xlabel("GWP (kg CO$_2$e per kg PS treated)")
ax.set_title(f"Net GWP {gwp0['Net GWP (kg CO2e/kg PS)']:.2f} kg CO$_2$e/kg PS; process only "
             f"{gwp0['Process GWP (kg CO2e/kg liquid product)']:.2f} kg CO$_2$e/kg product")
fig.savefig(os.path.join(OUT, "fig_sim_06_gwp_contributions.png"))
norm = imp.div(imp.abs().sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(4.3, 7.6))       # the long indicator names add ~2.9 in on the left
norm.plot(kind="barh", stacked=True, ax=ax, width=0.8, colormap="tab10")
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlabel("Share of total absolute impact (negative = credit)"); ax.set_xlim(-1, 1)
ax.legend(frameon=False, bbox_to_anchor=(0.3, -0.07), loc="upper center", ncol=2, fontsize=7.5)
ax.tick_params(axis="y", labelsize=8)
fig.savefig(os.path.join(OUT, "fig_sim_07_all_impact_categories.png"))

#%% TEA and LCA tornado (one-at-a-time)
def evaluate(system, **tea_kw):
    t = get_tea(system, **tea_kw); al = allocated_msps(t)[0]
    return {"MSP": mean_msp(t), "Styrene MSP": al["Styrene"], "BTX MSP": al["BTX"],
            "GWP": gwp_results(system)["Net GWP (kg CO2e/kg PS)"]}

base_eval = evaluate(ps_sys)
msp_b, gwp_b = base_eval["MSP"], base_eval["GWP"]
METRICS = {"MSP": "MSP", "Styrene MSP": "Styrene MSP", "BTX MSP": "BTX MSP", "GWP": "GWP"}


def torn_row(name, low, high, r_lo, r_hi):
    row = {"Parameter": name, "Low value": low, "High value": high}
    for k in METRICS:
        unit = "" if k == "GWP" else " ($/t)"
        row[f"{k} low{unit}"] = r_lo[k]; row[f"{k} high{unit}"] = r_hi[k]
    torn.append(row)
torn = []


def oat(name, setter, base, low, high, resimulate=True):
    res = []
    for v in (low, high):
        setter(v)
        s = run(create_ps_sys()) if resimulate else ps_sys
        res.append(evaluate(s))
    setter(base)
    torn_row(name, low, high, res[0], res[1])


def price_setter(k):
    return lambda v: prices.__setitem__(k, v)

# (product market prices do not change the mean or allocated MSPs - those are themselves prices - so they are
#  not varied; their effect shows in the "others at market" MSPs of the feed-price sensitivity)
for k in ["PS", "Fresh catalyst", "NG"]:
    oat(f"{k} price ($/kg)", price_setter(k), prices[k], prices[k] * 0.8, prices[k] * 1.2)
T_lo, T_hi = GP_DATA_RANGE[(A["regime"], A["catalyst"])]          # stay inside the GP data range
oat("Reactor temperature (°C)", lambda v: A.__setitem__("T_C", v), A["T_C"],
    max(A["T_C"] - 50, T_lo), min(A["T_C"] + 50, T_hi))
oat("Char share of non-liquid", lambda v: A.__setitem__("solid_share_nonliquid", v),
    A["solid_share_nonliquid"], 0.25, 0.75)
oat("Depolymerisation enthalpy (kJ/kg)", lambda v: A.__setitem__("dH_depolymerisation", v),
    A["dH_depolymerisation"], A["dH_depolymerisation"] * 0.8, A["dH_depolymerisation"] * 1.2)
# GP 95% interval of each yield, others at the mean
lo_hi = gp.interval(A["T_C"], A["regime"], A["catalyst"]); ymean = gp.raw(A["T_C"], A["regime"], A["catalyst"])[0]
for k in ["Liquid", "Styrene", "BTX", "Total aromatics"]:
    vals = []
    for v in lo_hi[k]:
        yy = dict(ymean); yy[k] = v
        vals.append(evaluate(run(create_ps_sys(yields=gp.consistent(yy)))))
    torn_row(f"{k} yield (GP 95% interval, wt%)", lo_hi[k][0], lo_hi[k][1], vals[0], vals[1])
ps_sys = run(create_ps_sys()); tea = get_tea(ps_sys)
# Lang factor (TEA only)
torn_row("Lang factor", 5.05 * 0.8, 5.05 * 1.2, evaluate(ps_sys, lang_factor=5.05 * 0.8),
         evaluate(ps_sys, lang_factor=5.05 * 1.2))
# impact factors (LCA only)
for row, name in [("PS_Feed", "Avoided PS treatment EF"), ("PSO", "Styrene EF"), ("AromaticsO", "Aromatics EF"),
                  ("Electricity", "Electricity EF"), ("boiler_turbogenerator_ng", "Natural gas EF")]:
    base = ef.loc[row, GWP]; r = []
    for f in (0.8, 1.2):
        ef.loc[row, GWP] = base * f; r.append(gwp_results(ps_sys)["Net GWP (kg CO2e/kg PS)"])
    ef.loc[row, GWP] = base
    torn_row(f"{name} (±20%)", base * 0.8, base * 1.2, {**base_eval, "GWP": r[0]}, {**base_eval, "GWP": r[1]})
tornado = pd.DataFrame(torn)


def plot_tornado(df, lo, hi, base, xlabel, fname):
    d = df.assign(span=(df[hi] - df[lo]).abs()).query("span > 1e-9").sort_values("span")
    fig, ax = plt.subplots(figsize=(6.4, 0.27 * len(d) + 1.0))
    y = np.arange(len(d))
    ax.barh(y, d[lo] - base, left=base, color=COL["green"], label="Low value")
    ax.barh(y, d[hi] - base, left=base, color=COL["blue"], label="High value")
    ax.axvline(base, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(d["Parameter"], fontsize=8.5)
    ax.set_xlabel(xlabel); ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(OUT, fname))

plot_tornado(tornado, "MSP low ($/t)", "MSP high ($/t)", msp_b, f"Mean liquid MSP (\\$/t); baseline \\${msp_b:,.0f}/t",
             "fig_sim_08_tornado_msp.png")
for p in MSP_PRODUCTS:
    plot_tornado(tornado, f"{p} MSP low ($/t)", f"{p} MSP high ($/t)", base_eval[f"{p} MSP"],
                 f"{p} MSP, market-value allocation (\\$/t); baseline \\${base_eval[p + ' MSP']:,.0f}/t",
                 f"fig_sim_08{'b' if p == 'Styrene' else 'c'}_tornado_{p.lower()}_msp.png")
plot_tornado(tornado, "GWP low", "GWP high", gwp_b, f"Net GWP (kg CO$_2$e/kg PS); baseline {gwp_b:.2f}",
             "fig_sim_09_tornado_gwp.png")
print(tornado.round(2).to_string(index=False))

#%% Monte Carlo: GP yield uncertainty -> MSP and GWP
N_MC = 300                                # increase for the paper (e.g. 1000)
rng = np.random.default_rng(42)
mc = []
t0 = time.time()
for i in range(N_MC):
    y = gp.sample(A["T_C"], A["regime"], A["catalyst"], rng, A["gp_sd_scale"])
    try:
        s = run(create_ps_sys(yields=y, ID="mc"))
        t = get_tea(s)
        al = allocated_msps(t)[0]; om = product_msps(t)
        mc.append({**{f"{k} (wt%)": v for k, v in y.items()}, "Products (t/yr)": product_flow_tpy(t),
                   "Mean MSP ($/t)": mean_msp(t),
                   **{f"{k} MSP, allocated ($/t)": al[k] for k in MSP_PRODUCTS},
                   **{f"{k} MSP, others at market ($/t)": om[k] for k in MSP_PRODUCTS},
                   "Break-even PS price ($/t)": breakeven_ps(t),
                   **gwp_results(s)})
    except Exception as e:
        mc.append({**{f"{k} (wt%)": v for k, v in y.items()}, "error": str(e)[:80]})
mc = pd.DataFrame(mc)
print(f"Monte Carlo: {N_MC} runs in {time.time() - t0:.0f} s, failed: {mc.get('error', pd.Series()).notna().sum()}")
if "error" in mc: print(mc["error"].value_counts().head())
ok = mc.dropna(subset=["Mean MSP ($/t)"])
summ = ok.describe(percentiles=[0.025, 0.05, 0.5, 0.95, 0.975]).T
print(summ[["mean", "2.5%", "50%", "97.5%"]].round(2))
fig, axs = plt.subplots(2, 2, figsize=(7, 5.2)); axs = axs.ravel()
for ax, col, c, base in [(axs[0], "Styrene MSP, allocated ($/t)", COL["blue"], base_eval["Styrene MSP"]),
                         (axs[1], "BTX MSP, allocated ($/t)", COL["gold"], base_eval["BTX MSP"]),
                         (axs[2], "Mean MSP ($/t)", COL["grey"], msp_b),
                         (axs[3], "Net GWP (kg CO2e/kg PS)", COL["orange"], gwp_b)]:
    hi_x = ok[col].quantile(0.99); n_out = (ok[col] > hi_x).sum()
    ax.hist(ok[col].clip(upper=hi_x), bins=30, color=c, alpha=0.85)
    if n_out: ax.text(0.98, 0.78, f"{n_out} runs > {hi_x:,.4g}\n(in last bin)", transform=ax.transAxes, ha="right", fontsize=7.5)
    q = ok[col].quantile([0.025, 0.5, 0.975])
    for v, ls in zip(q, [":", "-", ":"]): ax.axvline(v, color="k", lw=1, ls=ls)
    ax.axvline(base, color=COL["green"], lw=1.5, label="GP mean yields")
    ax.set_xlabel(col.replace("CO2e", "CO$_2$e").replace("$/t", "\\$/t")); ax.set_ylabel("Runs")
    ax.set_title(f"median {q.iloc[1]:,.4g} [95%: {q.iloc[0]:,.4g} to {q.iloc[2]:,.4g}]", fontsize=8.5)
axs[0].legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sim_10_monte_carlo.png"))
# Spearman rank correlation of yields with the outputs (which yield uncertainty matters most)
outs_ = ["Styrene MSP, allocated ($/t)", "BTX MSP, allocated ($/t)", "Mean MSP ($/t)", "Net GWP (kg CO2e/kg PS)"]
ins_ = [c for c in ok.columns if c.endswith("(wt%)")]
spear = ok[ins_ + outs_].rank().corr().loc[ins_, outs_]
print(spear.round(2))

#%% Monte Carlo: probability of profit vs waste PS price
# P(NPV > 0 at PS price p) = share of yield draws whose break-even PS price is above p
be_mc = ok["Break-even PS price ($/t)"].dropna().values
p_ax = np.linspace(-200, 1200, 141)
prob = np.array([(be_mc > p).mean() for p in p_ax])
prof_df = pd.DataFrame({"PS price ($/t)": p_ax, "Probability NPV > 0": prob})
q = np.percentile(be_mc, [5, 50, 95])
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(p_ax, prob * 100, lw=2, color=COL["blue"])
ax.axvspan(p_ax[0], 0, color=COL["grey"], alpha=0.12, lw=0)
ax.text(p_ax[0], 3, " gate fee", fontsize=8, color=COL["grey"])
for v, lab, yv in [(q[0], "95 % chance\nof profit", 95), (q[1], "50 %", 50), (q[2], "5 %", 5)]:
    ax.plot(v, yv, "o", color=COL["blue"], ms=6)
    ax.annotate(f"{lab}  \\${v:,.0f}/t", (v, yv), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
ax.axvline(prices["PS"] * 1000, color=COL["grey"], lw=0.8, ls="--")
ax.text(prices["PS"] * 1000, 12, " base case", va="bottom", fontsize=8, color=COL["grey"])
ax.set_ylim(0, 102); ax.set_xlabel("Waste PS price (\\$/t)")
ax.set_ylabel("Probability NPV > 0 (%)")
ax.set_title(f"{len(be_mc)} Monte Carlo runs; products at market price")
fig.savefig(os.path.join(OUT, "fig_sim_10b_probability_of_profit.png"))

#%% Save everything to Excel
ps_sys = run(create_ps_sys()); tea = get_tea(ps_sys)
summary = pd.DataFrame({
    "Item": ["Reactor temperature (°C)", "Catalyst class", "Capacity (t/d)",
             *[f"GP {k} yield (wt% of PS)" for k in pyro.last_yields],
             "Mass-balance closure error (%)", "Reactor duty (MJ/kg PS)", "NG to reactor (kg/kg PS)",
             "Installed equipment cost (MM$)", "FCI (MM$)", "TCI (MM$)", "AOC (MM$/yr)", "Labour (MM$/yr)",
             "Products (t/yr)", "Mean liquid MSP ($/t)",
             *[f"{k} MSP, market-value allocation ($/t)" for k in MSP_PRODUCTS],
             *[f"{k} MSP, others at market ($/t)" for k in MSP_PRODUCTS],
             *gwp0.keys(), "MC median MSP ($/t)", "MC 95% MSP low", "MC 95% MSP high",
             *[f"MC {k} MSP allocated, {q} ($/t)" for k in MSP_PRODUCTS for q in ("median", "95% low", "95% high")],
             "MC median net GWP", "MC 95% GWP low", "MC 95% GWP high"],
    "Value": [A["T_C"], A["catalyst"], A["capacity_tpd"], *pyro.last_yields.values(), err,
              pyro.duty / 1e3 / feedF, S(ps_sys, "pyrolyzer_ng").F_mass / feedF,
              tea.DPI / 1e6, tea.FCI / 1e6, tea.TCI / 1e6, tea.AOC / 1e6, tea.labor_cost / 1e6,
              product_flow_tpy(tea), msp0, *[alloc0[k] for k in MSP_PRODUCTS], *[msps0[k] for k in MSP_PRODUCTS],
              *gwp0.values(), *ok["Mean MSP ($/t)"].quantile([0.5, 0.025, 0.975]),
              *[v for k in MSP_PRODUCTS for v in ok[f"{k} MSP, allocated ($/t)"].quantile([0.5, 0.025, 0.975])],
              *ok["Net GWP (kg CO2e/kg PS)"].quantile([0.5, 0.025, 0.975])]})
with pd.ExcelWriter(os.path.join(OUT, "PS_simulation_GPR_results.xlsx")) as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    pd.DataFrame({"Parameter": list(A) + [f"Price {k} ($/kg)" for k in prices],
                  "Value": [str(v) for v in A.values()] + list(prices.values())}).to_excel(xw, sheet_name="Assumptions", index=False)
    stream_table.to_excel(xw, sheet_name="Stream table (kg per h)")
    msp_table.to_excel(xw, sheet_name="Product MSPs")
    pd.Series(breakdown, name="$/t liquid products").to_excel(xw, sheet_name="MSP breakdown")
    fp_df.to_excel(xw, sheet_name="Feed price sensitivity", index=False)
    ps_df.to_excel(xw, sheet_name="PS price - 3 product MSPs", index=False)
    npv_df.to_excel(xw, sheet_name="NPV map PS x styrene (MM$)")
    heat_df.to_excel(xw, sheet_name="Styrene MSP PS price x T")
    cap_df.to_excel(xw, sheet_name="Capacity sweep", index=False)
    prof_df.to_excel(xw, sheet_name="Probability of profit", index=False)
    sweep.to_excel(xw, sheet_name="Temperature sweep", index=False)
    pfd_table.to_excel(xw, sheet_name="PFD stream table", index=False)
    for k_, v_ in sankey_tables.items():
        v_.to_excel(xw, sheet_name=k_, index=False)
    eq.sort_values(ascending=False).rename("Purchase cost (MM$)").to_excel(xw, sheet_name="Equipment costs")
    imp.T.to_excel(xw, sheet_name="LCA per kg PS")
    tornado.to_excel(xw, sheet_name="Tornado", index=False)
    mc.to_excel(xw, sheet_name="Monte Carlo runs", index=False)
    summ.to_excel(xw, sheet_name="Monte Carlo summary")
    spear.to_excel(xw, sheet_name="MC rank correlations")
print("Saved to", OUT)
