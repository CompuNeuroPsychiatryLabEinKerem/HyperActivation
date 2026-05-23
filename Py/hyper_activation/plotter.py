"""Scatter plotter for HyperActivation dimensions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, t as t_dist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import yaml


DimensionName = Literal['Activation', 'OrientationPerformance', 'Connectivity', 'HippocampusVol', 'Diagnosis']


class RegressionStats(NamedTuple):
    """Statistics for C ~ X, C ~ Y, C ~ X + Y, C ~ X*Y regressions."""
    r_X: float
    p_X: float
    r_Y: float
    p_Y: float
    r_XY: float      # C ~ X + Y (adjusted)
    p_XY: float
    r_XxY: float     # C ~ X*Y interaction (adjusted)
    p_XxY: float


class ScatterResult(NamedTuple):
    """Result object from scatter plot."""
    subjects: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    C: np.ndarray
    Ctrimmed: np.ndarray
    Diag: np.ndarray
    fig: Figure | None


class LogitStats(NamedTuple):
    """Statistics for D ~ X, D ~ Y, D ~ X + Y, D ~ X*Y logistic regressions."""
    p_X: float
    p_Y: float
    p_XY: float
    p_XxY: float
    auc_X: float
    auc_Y: float
    auc_XY: float
    auc_XxY: float


def _looAuc(data: pd.DataFrame, formula: str, d: np.ndarray) -> float:
    """Compute leave-one-out AUC for a logistic regression formula."""
    n = len(d)
    data = data.reset_index(drop=True)
    loo_probs = np.empty(n)
    for i in range(n):
        train = data.drop(index=i).reset_index(drop=True)
        test = data.iloc[[i]].reset_index(drop=True)
        try:
            m = smf.logit(formula, train).fit(disp=0)
            loo_probs[i] = m.predict(test).iloc[0]
        except:
            loo_probs[i] = np.nan
    valid = ~np.isnan(loo_probs)
    if valid.sum() < len(d) * 0.9:
        return np.nan
    return roc_auc_score(d[valid], loo_probs[valid])


def _computeLogitStats(x: np.ndarray, y: np.ndarray, d: np.ndarray) -> LogitStats:
    """Compute logistic regression llr_pvalue and LOO-AUC for diagnosis prediction."""
    data = pd.DataFrame({'D': d, 'X': x, 'Y': y})

    def _fit(formula):
        try:
            m = smf.logit(formula, data).fit(disp=0)
            p = m.llr_pvalue
        except:
            p = np.nan
        try:
            auc = _looAuc(data, formula, d)
        except:
            auc = np.nan
        return p, auc

    p_X, auc_X = _fit('D ~ X')
    p_Y, auc_Y = _fit('D ~ Y')
    p_XY, auc_XY = _fit('D ~ X + Y')
    p_XxY, auc_XxY = _fit('D ~ X * Y')

    return LogitStats(p_X=p_X, p_Y=p_Y, p_XY=p_XY, p_XxY=p_XxY,
                      auc_X=auc_X, auc_Y=auc_Y, auc_XY=auc_XY, auc_XxY=auc_XxY)


def _computeStats(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> RegressionStats:
    """Compute regression statistics for C ~ X, C ~ Y, C ~ X + Y, C ~ X*Y."""
    # C ~ X
    X1 = sm.add_constant(x)
    m1 = sm.OLS(c, X1).fit()

    # C ~ Y
    X2 = sm.add_constant(y)
    m2 = sm.OLS(c, X2).fit()

    # C ~ X + Y
    X3 = sm.add_constant(np.column_stack([x, y]))
    m3 = sm.OLS(c, X3).fit()

    # C ~ X*Y (interaction: X + Y + X:Y)
    xy = x * y
    X4 = sm.add_constant(np.column_stack([x, y, xy]))
    m4 = sm.OLS(c, X4).fit()

    def toR(rsq):
        return np.sqrt(max(rsq, 0))

    return RegressionStats(
        r_X=toR(m1.rsquared), p_X=m1.f_pvalue,
        r_Y=toR(m2.rsquared), p_Y=m2.f_pvalue,
        r_XY=toR(m3.rsquared_adj), p_XY=m3.f_pvalue,
        r_XxY=toR(m4.rsquared_adj), p_XxY=m4.f_pvalue
    )


class HyperActivationPlotter:
    """
    Pre-computes corrected values for all dimensions and provides fast scatter plotting.

    Usage:
        plotter = HyperActivationPlotter()
        fig = plotter.scatter(X='Connectivity', Y='Activation', C='OrientationPerformance')
    """

    def __init__(self, projectDir: Path | str | None = None):
        self.projectDir = Path(projectDir or r"C:\Projects\HyperActivation")

        # Load config
        with open(self.projectDir / "Config.yml") as f:
            self.cfg = yaml.safe_load(f)

        # Load data
        with open(self.projectDir / self.cfg['DATA_SOURCES']['BOLD_And_Behavioral'], 'rb') as f:
            self.df: pd.DataFrame = pickle.load(f)

        # Load null models for OrientationPerformance (both versions)
        self.nullModels = {}
        for key in ['All', 'CN_Only']:
            path = self.projectDir / self.cfg['DATA_SOURCES']['Null_Models_OrientationPerformance'][key]
            with open(path, 'rb') as f:
                self.nullModels[key] = pickle.load(f)

        self.rspSize = self.cfg['MISC']['Rsp_Size']

        # Extract config values
        self.refAge = self.cfg['MISC']['Reference_Age_For_Correction']
        self.refSex = self.cfg['MISC']['Reference_Sex_For_Correction']
        self.colorK = self.cfg['MISC']['ColorNormlization']
        self.connMaxVal = self.cfg['MISC']['ConnectivityMaxVal']
        self.defaultCfg = self.cfg['DEFAULT_CFG']
        corrMatCfg = self.defaultCfg.get('CorrelationMatrix', {})
        self.corrMatDo = corrMatCfg.get('Do', False)
        self.corrMatAttrs = [a.strip() for a in corrMatCfg.get('Attrs', '').split(',') if a.strip()] if isinstance(corrMatCfg.get('Attrs'), str) else corrMatCfg.get('Attrs', [])
        self.corrMatCmap = corrMatCfg.get('Colormap', 'cool')
        self.aliases = self.defaultCfg.get('Aliases', {})
        # AUC bar plot: list of entries, each a comma-separated string of variable names
        aucCfg = self.defaultCfg.get('AUC_BarPlot', [])
        self.aucBarPlotGroups = [[v.strip() for v in entry.split(',')] for entry in aucCfg] if aucCfg else []
        self.hippoCoefs = self.cfg['NULL_MODELS']['HippoVol_Null_Model']
        self.hippoOffsets = self.hippoCoefs['Offsets']

        # Marker size mapping (scaled by MarkerScale)
        scale = self.cfg['MISC'].get('MarkerScale', 1)
        self.sizeMap = {'small': 20 * scale, 'large': 60 * scale}
        self.defaultSize = 40 * scale

        # Pre-compute corrected values
        self._precompute()

    def _precompute(self):
        """Pre-compute corrected values for all dimension configurations."""
        df = self.df

        # Demographics
        self.age = df['DMG_Age'].values
        self.isMale = (df['DMG_Sex'] == 'M').astype(float).values

        # eTIV from configured column (for HippoVol correction)
        eTIVcol = self.hippoCoefs.get('Columns', {}).get('eTIV')
        self.eTIV = df[eTIVcol].values if eTIVcol and eTIVcol in df.columns else None

        # Binary diagnosis: CTRL or CN = 0 (healthy), REST = 1 (impaired)
        diags = df['Diags'].values if 'Diags' in df.columns else None
        dg1 = df['Dg1'].values if 'Dg1' in df.columns else None
        self.diagBinary = np.ones(len(df))  # Default to impaired
        if diags is not None:
            self.diagBinary[diags == 'CTRL'] = 0
        if dg1 is not None:
            self.diagBinary[dg1 == 'CN'] = 0

        # Connectivity: clipped mean per ParcelType
        self.connectivity = {}
        for parcelType in ['Singles', 'Bilaterals', 'Networks', 'All']:
            if parcelType == 'All':
                connCols = [c for c in df.columns if c.startswith('RSP_TO_')]
            else:
                connCols = [c for c in df.columns if c.startswith(f'RSP_TO_{parcelType}')]
            if connCols:
                connVals = df[connCols].values
                self.connectivity[parcelType] = np.clip(connVals, -self.connMaxVal, self.connMaxVal).mean(1)

        # HippocampusVol: pre-compute for L, R, Sum with all correction types
        self.hippoVol = {}
        for hemi in ['L', 'R']:
            self.hippoVol[hemi] = {}
            coefs = self.hippoCoefs[hemi]
            for measure in ['CFT', 'CFT2', 'CFT2_MNI', 'CFT_MNI', 'FS']:
                colName = f"HPVOL_{measure}_{hemi}"
                if colName not in df.columns:
                    continue
                raw = df[colName].values
                corrected = self._correctHippo(raw, coefs)  # Returns dict with Abs, Rel, LogRel
                self.hippoVol[hemi][measure] = corrected

        # Compute Sum (L+R) by summing raw data first, then applying corrections
        self.hippoVol['Sum'] = {}
        for measure in self.hippoVol['L']:
            if measure in self.hippoVol['R']:
                L = self.hippoVol['L'][measure]
                R = self.hippoVol['R'][measure]
                rawSum = L['_raw'] + R['_raw']
                predSum = L['_predActual'] + R['_predActual']
                predRefSum = L['_predRef'] + R['_predRef']
                ratio = rawSum / (predSum + 1e-10)
                residuals = rawSum - predSum
                zScored = (residuals - residuals.mean()) / (residuals.std() + 1e-10)
                self.hippoVol['Sum'][measure] = {
                    'Abs': predRefSum + residuals,
                    'Rel': ratio,
                    'LogRel': np.log2(ratio + 1e-10),
                    'Z': zScored,
                }

        # OrientationPerformance: corrected using null models (both All and CN_Only)
        self.performance = {}
        for nullModelKey in ['All', 'CN_Only']:
            self.performance[nullModelKey] = {}
            for domain in ['Space', 'Time', 'Person', 'Lex']:
                self.performance[nullModelKey][domain] = {}
                for subtype in ['', 'D1', 'D2']:
                    colName = f"EC_{domain}" if subtype == '' else f"EC_{domain}_{subtype}"
                    if colName not in df.columns:
                        continue
                    raw = df[colName].values
                    corrected = self._correctPerformance(raw, domain, nullModelKey)
                    self.performance[nullModelKey][domain][subtype] = corrected

        # Activation: no correction needed, just cache column access patterns
        self.activation = {}  # Lazy load on demand

    def _correctHippo(self, raw: np.ndarray, coefs: Dict) -> Dict[str, np.ndarray]:
        """Apply polynomial correction for hippocampus volume. Returns dict with Abs, Rel, LogRel."""
        age, refAge = self.age, self.refAge
        eTIVref = self.hippoOffsets['eTIV']
        ageOff = self.hippoOffsets['Age']

        # Centered eTIV
        rawETIV = self.eTIV if self.eTIV is not None else np.full(len(age), eTIVref)
        e = rawETIV - eTIVref
        eRef = 0.0

        # Centered age
        a = age - ageOff
        aRef = refAge - ageOff

        # Prediction: Intercept + Age*a + Age2*a² + Age3*a³ + eTIV*e [+ eTIV2*e²]
        def pred(a_, e_):
            p = (coefs['Intercept'] +
                 coefs['Age'] * a_ +
                 coefs['Age2'] * a_**2 +
                 coefs['Age3'] * a_**3 +
                 coefs['eTIV'] * e_)
            if 'eTIV2' in coefs:
                p = p + coefs['eTIV2'] * e_**2
            return p

        predRef = pred(aRef, eRef)
        predActual = pred(a, e)

        ratio = raw / (predActual + 1e-10)
        residuals = raw - predActual
        zScored = (residuals - residuals.mean()) / (residuals.std() + 1e-10)
        return {
            'Abs': predRef + residuals,
            'Rel': ratio,
            'LogRel': np.log2(ratio + 1e-10),
            'Z': zScored,
            '_raw': raw,
            '_predActual': predActual,
            '_predRef': predRef,
        }

    def _correctPerformance(self, raw: np.ndarray, domain: str, nullModelKey: str) -> np.ndarray:
        """Apply OLS correction for orientation performance."""
        models = self.nullModels[nullModelKey]
        if domain not in models or domain == 'Description':
            return raw  # No model available, return raw

        model = models[domain]

        # Build prediction DataFrame
        refIsMale = 0.0 if self.refSex == 'F' else 1.0

        dfPred = pd.DataFrame({'Age': self.age, 'isMale': self.isMale})
        dfRef = pd.DataFrame({'Age': [self.refAge] * len(self.age), 'isMale': [refIsMale] * len(self.age)})

        predActual = model.predict(dfPred)
        predRef = model.predict(dfRef)

        # Corrected = pred(ref) + (raw - pred(actual))
        return predRef + (raw - predActual)

    def _getActivation(self, domain: str, signalType: str, region: str, size: str) -> np.ndarray:
        """Get activation values as mean (no correction)."""
        key = (domain, signalType, region, size)
        if key not in self.activation:
            colName = f"{domain}_{signalType}_{region}_{size}"
            vals = self.df[colName].values
            # Convert sum to mean
            if size == 'Sum':
                vals = vals / self.rspSize
            elif size.startswith('Max_'):
                k = int(size.split('_')[1])
                vals = vals / k
            self.activation[key] = vals
        return self.activation[key]

    def _getDimensionValues(self, dim: DimensionName, opts: Dict[str, Any]) -> np.ndarray:
        """Get corrected values for a dimension given options."""
        if dim == 'Activation':
            cfg = {**self.defaultCfg['Activation'], **opts.get('Activation', {})}
            return self._getActivation(cfg['Domains'], cfg['SignalType'], cfg['Region'], cfg['Size'])

        elif dim == 'OrientationPerformance':
            cfg = {**self.defaultCfg['OrientationPerformance'], **opts.get('OrientationPerformance', {})}
            nullModelKey = cfg.get('Null_Model', 'All')
            domain = cfg['Domain']
            subtype = cfg.get('Subtype', '') or ''
            return self.performance[nullModelKey][domain][subtype]

        elif dim == 'Connectivity':
            cfg = {**self.defaultCfg['Connectivity'], **opts.get('Connectivity', {})}
            parcelType = cfg.get('ParcelType', 'All')
            return self.connectivity[parcelType]

        elif dim == 'HippocampusVol':
            cfg = {**self.defaultCfg['HippocampusVol'], **opts.get('HippocampusVol', {})}
            corrType = cfg.get('CorrectionType', 'Abs')
            return self.hippoVol[cfg['Hemi']][cfg['Measure']][corrType]

        elif dim == 'Diagnosis':
            return self.diagBinary

        elif dim in self.df.columns:
            return self.df[dim].values

        raise ValueError(f"Unknown dimension: {dim}")

    def _normalizeColor(self, c: np.ndarray) -> np.ndarray:
        """Normalize values to [-1, 1] using z-score, clip at ±K std, scale."""
        mu, sigma = c.mean(), c.std()
        z = (c - mu) / (sigma + 1e-10)

        # Clip at ±K std, then scale to [-1, 1]
        clipped = np.clip(z, -self.colorK, self.colorK)
        return clipped / self.colorK

    def _toRGB(self, cNorm: np.ndarray) -> np.ndarray:
        """Convert normalized [-1,1] values to RGB (Red -> Violet -> Blue)."""
        a = (cNorm + 1) / 2  # Map to [0, 1]
        r = 1 - a
        g = np.zeros_like(a)
        b = a
        return np.column_stack([r, g, b])

    def _alias(self, name: str) -> str:
        """Return the display alias for a dimension name, or the name itself."""
        return self.aliases.get(name, name)

    def _getMarkerSizes(self, options: Dict[str, Any]) -> np.ndarray | float:
        """Get marker sizes based on diagnosis config."""
        diagCfg = {**self.defaultCfg['UseDiagnoses'], **options.get('UseDiagnoses', {})}

        if not diagCfg['Bool']:
            return self.defaultSize

        col = diagCfg['Columns']
        key = diagCfg['Key']
        diags = self.df[col].values
        sizes = np.array([self.sizeMap.get(key.get(d, 'small'), self.defaultSize) for d in diags])
        return sizes

    def _drawCorrMatrices(self, axCorr, axPartial, options: Dict[str, Any]):
        """Draw correlation and partial-correlation heatmaps for configured attributes."""
        attrs = self.corrMatAttrs
        n = len(attrs)
        nObs = len(self.age)
        vals = [self._getDimensionValues(a, options) for a in attrs]
        nComparisons = n * (n - 1) // 2  # Bonferroni correction

        # Full correlation matrix (r and p)
        corrMat = np.zeros((n, n))
        corrP = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, p = pearsonr(vals[i], vals[j])
                corrMat[i, j] = corrMat[j, i] = r
                corrP[i, j] = corrP[j, i] = p

        # Partial correlation matrix (r and p via t-test)
        partialMat = np.zeros((n, n))
        partialP = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                others = [vals[k] for k in range(n) if k != i and k != j]
                control = np.column_stack(others) if others else None
                if control is not None:
                    k = control.shape[1]
                    reg_a = LinearRegression().fit(control, vals[i])
                    reg_b = LinearRegression().fit(control, vals[j])
                    res_a = vals[i] - reg_a.predict(control)
                    res_b = vals[j] - reg_b.predict(control)
                    r, _ = pearsonr(res_a, res_b)
                    # t-test for partial correlation: df = n - 2 - k
                    df = nObs - 2 - k
                    t_val = r * np.sqrt(df / (1 - r**2 + 1e-10))
                    p = 2 * t_dist.sf(np.abs(t_val), df)
                else:
                    r, p = pearsonr(vals[i], vals[j])
                partialMat[i, j] = partialMat[j, i] = r
                partialP[i, j] = partialP[j, i] = p

        def stars(p):
            """Bonferroni-corrected significance stars."""
            adj = p * nComparisons
            if adj < 0.001:
                return '***'
            elif adj < 0.01:
                return '**'
            elif adj < 0.05:
                return '*'
            return ''

        shortLabels = [self._alias(a) for a in attrs]

        # Mask: show only lower triangle without diagonal
        mask = np.triu(np.ones((n, n), dtype=bool))

        for ax, mat, pMat, title in [(axCorr, corrMat, corrP, 'Correlation'),
                                      (axPartial, partialMat, partialP, 'Partial Correlation')]:
            dispMat = np.abs(mat).copy()
            dispMat[mask] = np.nan
            im = ax.imshow(dispMat, cmap=self.corrMatCmap, vmin=0, vmax=0.6, aspect='equal')
            # Crop to lower triangle: hide diagonal and above
            ax.set_xlim(-0.5, n - 1.5)
            ax.set_ylim(n - 0.5, 0.5)
            # Tick labels: x on columns 0..n-2, y on rows 1..n-1
            ax.set_xticks(range(n - 1))
            ax.set_yticks(range(1, n))
            ax.set_xticklabels(shortLabels[:n - 1], rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(shortLabels[1:], fontsize=7)
            ax.set_title(title, fontsize=9)
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(top=False, right=False)
            for i in range(n):
                for j in range(n):
                    if mask[i, j]:
                        continue
                    v = dispMat[i, j]
                    s = stars(pMat[i, j])
                    color = 'white' if v > 0.4 else 'black'
                    ax.text(j, i, f'{v:.2f}{s}', ha='center', va='center',
                            color=color, fontsize=7, fontweight='bold')
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _drawAucBarPlot(self, axBar, options: Dict[str, Any]):
        """Draw LOO-AUC bar plot for configured variable groups."""
        groups = self.aucBarPlotGroups
        if not groups:
            axBar.axis('off')
            return

        # Color by number of variables: 1->deepskyblue, 2->royalblue, 3+->mediumblue
        colorMap = {1: 'deepskyblue', 2: 'royalblue'}
        defaultColor = 'mediumblue'

        labels = []
        aucs = []
        barColors = []
        d = self.diagBinary

        for group in groups:
            vals = [self._getDimensionValues(v, options) for v in group]
            nVars = len(group)
            # Build additive formula: D ~ V0 + V1 + ...
            data = pd.DataFrame({'D': d})
            varNames = []
            for k, v in enumerate(vals):
                vName = f'V{k}'
                data[vName] = v
                varNames.append(vName)
            formula = 'D ~ ' + ' + '.join(varNames)
            try:
                auc = _looAuc(data, formula, d)
            except:
                auc = np.nan
            aliased = [self._alias(v) for v in group]
            labels.append(' + '.join(aliased))
            aucs.append(auc)
            barColors.append(colorMap.get(nVars, defaultColor))

        x = np.arange(len(labels))
        axBar.bar(x, aucs, color=barColors, edgecolor='none', width=0.8)
        axBar.set_xticks(x)
        axBar.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        axBar.set_ylabel('LOO AUC', fontsize=8)
        axBar.set_ylim(0.5, max(0.85, max(a for a in aucs if np.isfinite(a)) + 0.05))
        axBar.axhline(0.5, color='grey', linewidth=0.5, linestyle='--')
        axBar.set_title('Logit LOO-AUC', fontsize=9)
        axBar.spines['top'].set_visible(False)
        axBar.spines['right'].set_visible(False)
        # Add value labels on bars
        for i, auc in enumerate(aucs):
            if np.isfinite(auc):
                axBar.text(i, auc + 0.01, f'{auc:.2f}', ha='center', va='bottom', fontsize=7)

    def scatter(
        self,
        X: DimensionName = 'Connectivity',
        Y: DimensionName = 'Activation',
        C: DimensionName = 'OrientationPerformance',
        ax: plt.Axes | None = None,
        show: bool = False,
        **options
    ) -> ScatterResult:
        """
        Create scatter plot with X, Y axes and color dimension.

        Args:
            X: Dimension for X-axis
            Y: Dimension for Y-axis
            C: Dimension for color
            ax: Optional matplotlib axes to plot on
            show: If True, display in independent window (uses TkAgg backend)
            **options: Override DEFAULT_CFG per dimension
                e.g., Activation={'Domain': 'Time'}, HippocampusVol={'Hemi': 'L'}
                      UseDiagnoses={'Bool': False} to disable size coding

        Returns:
            ScatterResult with X, Y, C, Ctrimmed arrays and fig (None if show=True)
        """

        # Suppress inline display when showing in window
        if show:
            plt.ioff()

        xVals = self._getDimensionValues(X, options)
        yVals = self._getDimensionValues(Y, options)
        isDiagColor = (C == 'Diagnosis')

        if isDiagColor:
            cVals = self.diagBinary
            cNorm = self.diagBinary
            # Blue for healthy (0), red for impaired (1)
            colors = np.array([[0, 0, 1] if d == 0 else [1, 0, 0] for d in self.diagBinary])
            sizes = self.defaultSize
        else:
            cVals = self._getDimensionValues(C, options)
            cNorm = self._normalizeColor(cVals)
            colors = self._toRGB(cNorm)
            sizes = self._getMarkerSizes(options)

        # Compute statistics
        logitStats = _computeLogitStats(xVals, yVals, self.diagBinary)
        olsStats = None if isDiagColor else _computeStats(xVals, yVals, cNorm)

        # Create figure with main plot, stats area, and optional right pane
        showCorrMat = self.corrMatDo and len(self.corrMatAttrs) >= 2
        showAucBar = bool(self.aucBarPlotGroups)
        showRightPane = showCorrMat or showAucBar
        if ax is None:
            if showRightPane:
                fig = plt.figure(figsize=(14, 10))
                gsOuter = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)
                gsLeft = gsOuter[0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.3)
                # Right pane: corr, partial, bar plot
                rightParts = []
                if showCorrMat:
                    rightParts += [1, 1]   # corr + partial
                if showAucBar:
                    rightParts += [1]      # bar plot
                gsRight = gsOuter[1].subgridspec(len(rightParts), 1,
                                                  height_ratios=rightParts, hspace=0.5)
                ax = fig.add_subplot(gsLeft[0])
                axStats = fig.add_subplot(gsLeft[1])
                ridx = 0
                if showCorrMat:
                    axCorr = fig.add_subplot(gsRight[ridx]); ridx += 1
                    axPartial = fig.add_subplot(gsRight[ridx]); ridx += 1
                else:
                    axCorr = axPartial = None
                axAucBar = fig.add_subplot(gsRight[ridx]) if showAucBar else None
            else:
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.3)
                ax = fig.add_subplot(gs[0])
                axStats = fig.add_subplot(gs[1])
                axCorr = axPartial = axAucBar = None
        else:
            fig = ax.figure
            axStats = None
            axCorr = axPartial = axAucBar = None

        ax.scatter(xVals, yVals, c=colors, s=sizes, alpha=0.7, edgecolors='none')

        xLabel = self._alias(X)
        yLabel = self._alias(Y)
        cLabel = self._alias(C)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_title(f'{yLabel} vs {xLabel} (color: {cLabel})')

        if isDiagColor:
            pass  # No legend; blue=healthy, red=impaired
        else:
            # Add colorbar with original C scale (Red -> Violet -> Blue)
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('RedVioletBlue', [(1,0,0), (0,0,1)])
            cMin, cMax = cVals.min(), cVals.max()
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cMin, cMax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(cLabel)

        # Format p-values
        def fmtP(p):
            return f'{p:.4f}' if p >= 0.0001 else f'{p:.2e}'

        # Compute pearsonr(X, Y)
        rXY, pXY = pearsonr(xVals, yVals)

        # Build stats text
        if isDiagColor:
            olsTxt = f"r(X,Y): r={rXY:.3f}, p={fmtP(pXY)}"
        else:
            olsTxt = (
                f"r(X,Y): r={rXY:.3f}, p={fmtP(pXY)}\n"
                f"OLS (C~):\n"
                f"  X:   R={olsStats.r_X:.3f}, p={fmtP(olsStats.p_X)}\n"
                f"  Y:   R={olsStats.r_Y:.3f}, p={fmtP(olsStats.p_Y)}\n"
                f"  X+Y: R={olsStats.r_XY:.3f}, p={fmtP(olsStats.p_XY)}\n"
                f"  X*Y: R={olsStats.r_XxY:.3f}, p={fmtP(olsStats.p_XxY)}"
            )
        logitTxt = (
            f"Logit (D~):\n"
            f"  X:   p={fmtP(logitStats.p_X)}, AUC={logitStats.auc_X:.3f}\n"
            f"  Y:   p={fmtP(logitStats.p_Y)}, AUC={logitStats.auc_Y:.3f}\n"
            f"  X+Y: p={fmtP(logitStats.p_XY)}, AUC={logitStats.auc_XY:.3f}\n"
            f"  X*Y: p={fmtP(logitStats.p_XxY)}, AUC={logitStats.auc_XxY:.3f}"
        )

        # Display stats below plot
        if axStats is not None:
            axStats.axis('off')
            axStats.text(0.05, 0.5, olsTxt, transform=axStats.transAxes, fontsize=9,
                         verticalalignment='center', fontfamily='monospace')
            axStats.text(0.55, 0.5, logitTxt, transform=axStats.transAxes, fontsize=9,
                         verticalalignment='center', fontfamily='monospace')

        # Draw correlation matrices in right pane
        if axCorr is not None and axPartial is not None:
            self._drawCorrMatrices(axCorr, axPartial, options)

        # Draw AUC bar plot in right pane
        if axAucBar is not None:
            self._drawAucBarPlot(axAucBar, options)

        result = ScatterResult(subjects=self.df.index.values, X=xVals, Y=yVals, C=cVals, Ctrimmed=cNorm, Diag=self.diagBinary, fig=fig)

        if show:
            plt.show(block=False)
            plt.ion()  # Restore interactive mode
            return ScatterResult(subjects=self.df.index.values, X=xVals, Y=yVals, C=cVals, Ctrimmed=cNorm, Diag=self.diagBinary, fig=None)

        return result

    def scatterTriple(
        self,
        X: DimensionName = 'Connectivity',
        Y: DimensionName = 'Activation',
        C: DimensionName = 'OrientationPerformance',
        show: bool = False,
        **options
    ) -> ScatterResult:
        """
        Create three side-by-side scatter plots: X-C, Y-C, Predicted-C.

        Args:
            X: Dimension for X
            Y: Dimension for Y
            C: Dimension for C (color in main scatter, Y-axis here)
            show: If True, display in independent window
            **options: Override DEFAULT_CFG per dimension

        Returns:
            ScatterResult with X, Y, C, Ctrimmed arrays and fig
        """
        if show:
            plt.ioff()

        xVals = self._getDimensionValues(X, options)
        yVals = self._getDimensionValues(Y, options)
        cVals = self._getDimensionValues(C, options)
        cNorm = self._normalizeColor(cVals)

        # Diagnosis colors
        diagCfg = {**self.defaultCfg['UseDiagnoses'], **options.get('UseDiagnoses', {})}
        if diagCfg['Bool']:
            diagColors = np.array([[0, 0, 1] if d == 0 else [1, 0, 0] for d in self.diagBinary])
        else:
            diagColors = self._toRGB(cNorm)

        # Fit model C ~ X*Y and get predictions
        data = pd.DataFrame({'C': cNorm, 'X': xVals, 'Y': yVals})
        model = sm.OLS.from_formula('C ~ X * Y', data).fit()
        cPred = model.predict(data)

        # Stats for each plot
        # Plot 1: C ~ X
        m1 = sm.OLS(cNorm, sm.add_constant(xVals)).fit()
        r1, p1 = np.sqrt(max(m1.rsquared, 0)), m1.f_pvalue

        # Plot 2: C ~ Y
        m2 = sm.OLS(cNorm, sm.add_constant(yVals)).fit()
        r2, p2 = np.sqrt(max(m2.rsquared, 0)), m2.f_pvalue

        # Plot 3: C ~ X*Y (model already fit)
        r3, p3 = np.sqrt(max(model.rsquared_adj, 0)), model.f_pvalue

        # Create figure
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1], hspace=0.4, wspace=0.3)

        def fmtP(p):
            return f'{p:.4f}' if p >= 0.0001 else f'{p:.2e}'

        # Plot 1: X vs C
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(xVals, cNorm, c=diagColors, s=self.defaultSize, alpha=0.7, edgecolors='none')
        ax1.set_xlabel(X)
        ax1.set_ylabel(C)
        ax1.set_title(f'{C} vs {X}')

        axS1 = fig.add_subplot(gs[1, 0])
        axS1.axis('off')
        axS1.text(0.5, 0.5, f'R={r1:.3f}, p={fmtP(p1)}', transform=axS1.transAxes,
                  fontsize=10, ha='center', va='center', fontfamily='monospace')

        # Plot 2: Y vs C
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(yVals, cNorm, c=diagColors, s=self.defaultSize, alpha=0.7, edgecolors='none')
        ax2.set_xlabel(Y)
        ax2.set_ylabel(C)
        ax2.set_title(f'{C} vs {Y}')

        axS2 = fig.add_subplot(gs[1, 1])
        axS2.axis('off')
        axS2.text(0.5, 0.5, f'R={r2:.3f}, p={fmtP(p2)}', transform=axS2.transAxes,
                  fontsize=10, ha='center', va='center', fontfamily='monospace')

        # Plot 3: Predicted vs Actual C
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(cPred, cNorm, c=diagColors, s=self.defaultSize, alpha=0.7, edgecolors='none')
        ax3.set_xlabel(f'Predicted {C}')
        ax3.set_ylabel(f'Actual {C}')
        ax3.set_title(f'{C}: Predicted vs Actual')
        # Add diagonal line
        lims = [min(cPred.min(), cNorm.min()), max(cPred.max(), cNorm.max())]
        ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

        axS3 = fig.add_subplot(gs[1, 2])
        axS3.axis('off')
        axS3.text(0.5, 0.5, f'R={r3:.3f}, p={fmtP(p3)}', transform=axS3.transAxes,
                  fontsize=10, ha='center', va='center', fontfamily='monospace')

        result = ScatterResult(subjects=self.df.index.values, X=xVals, Y=yVals, C=cVals, Ctrimmed=cNorm, Diag=self.diagBinary, fig=fig)

        if show:
            plt.show(block=False)
            plt.ion()
            return ScatterResult(subjects=self.df.index.values, X=xVals, Y=yVals, C=cVals, Ctrimmed=cNorm, Diag=self.diagBinary, fig=None)

        return result
