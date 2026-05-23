# Coding Patterns — uri's style

Extracted from AssutaCleanRest.py / AssutaCleanBold.py.
Use this as a reference when refactoring or writing new modules in this codebase.

---

## Imports & aliases

Shorten heavy namespaces at the top so every callsite stays terse:

```python
import os, re, yaml
op  = os.path
opj = op.join
from collections import defaultdict as dd
from copy import deepcopy
from sklearn.linear_model import LinearRegression as linreg
```

Group stdlib / third-party / local in blocks separated by a blank line (no comments needed).

---

## Module-level lambdas for tiny transforms

One-liner math or predicate that doesn't deserve a `def`:

```python
r2z    = lambda x: np.arctanh(np.clip(x, -(1-1e-6), 1-1e-6))
isReg  = lambda txt: not re.match(r'\A\w+\Z', txt)   # is txt a regex pattern?
```

Keep them at module level, not buried inside functions, so they read like named constants.

---

## Conditional suffix / label in one line

```python
fsuff = f'_fwhm{fwhm}' if fwhm else ''
wn, btype = (band, 'bandpass') if band[-1] < 1/(2*TR) else (band[0], 'highpass')
```

Ternary for value selection; always both branches on one line when they fit.

---

## Boolean flags derived from parameters

Turn numeric thresholds into clear intent flags immediately:

```python
doScrubFD   = fdThr   > 0
doScrubDVars = dvarsThr > 0
```

Then use `if doScrubFD:` throughout — never re-test the raw threshold.

---

## In-place operators on arrays

Prefer in-place where semantics allow; avoids temp variable and signals mutation:

```python
isOutlier |= confoundsDf.framewise_displacement.values > fdThr
boldData  -= regobj.predict(regressors)
```

---

## Set arithmetic for subject matching

```python
boldSet  = set(boldfilesDict)
cnfdSet  = set(cnfdfilesDict)
common   = boldSet & cnfdSet
onlyBold = sorted(boldSet - cnfdSet)
onlyConf = sorted(cnfdSet - boldSet)
```

Report mismatches with `if onlyBold: print(...)` — no else needed.

---

## Separate technicalities from function

Technical details (getting a regex right, parsing a path) should be isolated and named
so the functional line reads as pure intent. Mess is acceptable — as long as it's contained:

```python
regBold   = re.compile(r'(\w{4})\b.*\-(\d)\-')               # Tech 1: correct regex
getBoldId = lambda path: regBold.match(op.split(path)[-1]).groups()  # Tech 2: id from path
boldDict  = {getBoldId(f): f for f in boldfiles}              # function: clear as a sentence
```

The functional line should be readable without knowing anything about what happens inside
`getBoldId`. The reader can trust the name and move on.

---

## Named-dict kwargs for long call signatures

Collect keyword args into a dict, then splat — keeps the callsite readable:

```python
args = dict(analog=False, fs=1/TR, output='sos', Wn=wn, btype=btype)
sos  = iirfilter(order, **args)
```

---

## Multi-return: cap at 3, then use a dict

Up to 3 return values is fine to unpack directly:

```python
return boldData, regressors, nRemoved   # fine
a, b, c = f()                           # fine
```

4+ values is a hydra — return a dict instead and unpack what you need:

```python
# definition
return dict(iOutliers=iOutliers, iInliers=iInliers,
            nRemoveHead=nRemoveHead, nRemoveTail=nRemoveTail,
            areEnoughInliers=areEnoughInliers)

# callsite — take only what's needed, or unpack all at once
out = getOutliers(...)
iOutliers, iInliers = out['iOutliers'], out['iInliers']
```

A named dict makes the callsite self-documenting and survives future additions without
breaking existing callers.

---

## Structured options dict with deep-merge helper

```python
DEFAULT_OPTS = dict(
    MISC   = dict(TR=2.02, remove1stVols=2),
    SCRUB  = dict(Do=False, fdThr=-1, dvarsThr=75, minStreakThr=-1, acceptThr=-1),
    REGRESS= dict(Do=True,  useConfounds=USE_CONFOUNDS),
    FILTER = dict(Do=True,  iirOrder=3, band=[0.008, 0.25]),
)

def getOpts(optsInp={}):
    opts = deepcopy(DEFAULT_OPTS)
    for key, subopts in optsInp.items():
        opts[key].update(subopts)
    return opts
```

Caller passes only overrides; function merges on top of defaults.
`deepcopy` prevents callers from poisoning the module-level dict.

---

## For-loop variable extraction

Unpack all loop variables in the `for` line — never index inside the body:

```python
for key, subopts in optsInp.items():          # dict
for hm, num in idxs.items():                  # nested dict
for cnt, (name, idxs) in enumerate(parcels.items()):  # enumerate + nested tuple
for ax, data, thr in zip([ax1,ax2], datasets, thresholds):  # zip multiple
```

---

## Skip-if-exists guard

```python
if os.path.exists(outFile):
    continue
```

Placed immediately before the expensive call — no nesting needed.

---

## `defaultdict` aliased as `dd`

```python
parcels    = dd(dict)
yeoAreaIdxs = dd(list)
```

Use `dd` wherever a plain dict would require an `if key not in` guard.

---

## Compact numpy idioms

```python
iInliers  = np.argwhere(~isOutlier).flatten()
iOutliers = np.argwhere( isOutlier).flatten()
padded    = np.hstack((1, isOutlier, 1))           # scalar + array concat
iNewVols  = np.hstack(tuple(np.arange(a+1, b) for a, b in pairs)) - 1
```

Chain `.flatten()` directly on `argwhere` — don't store the 2-D result.

---

## Object methods over module functions

Prefer array/series methods over their `np.*` equivalents — reads left-to-right and
keeps the subject of the expression up front:

```python
# preferred
varx = ((x - x.mean()) ** 2).mean()
mu   = x.mean(axis=0)
s    = x.std()

# avoid
varx = np.mean((x - np.mean(x)) ** 2)
mu   = np.mean(x, axis=0)
```

Same principle for linear algebra — use operators over functions:

```python
Z = X @ Y.T          # preferred
Z = np.dot(X, np.transpose(Y))   # avoid
```

---

## Regex-driven column selection

Treat confound names as either plain strings or regex patterns; resolve at runtime:

```python
isReg = lambda txt: not re.match(r'\A\w+\Z', txt)

for name in useRegressors:
    if isReg(name):
        chosenCols += [col for col in allCols if re.compile(name).match(col)]
    elif name in allCols:
        chosenCols.append(name)
```

---

## `opj` for every path construction

Never concatenate strings for paths; `opj` everywhere:

```python
boldFile = opj(bolddir, f'{sbj}-rest-1-cifti_sbj_Atlas_s0{fsuff}.dtseries.nii')
outFile  = opj(basedir, 'Task_Half_Cleaned', 'CleanCfg.yaml')
```

---

## Output filename derived from input

Strip known suffix, append tag, re-attach suffix — no separate output dir variable needed:

```python
sfx     = '.dtseries.nii'
outFile = boldFile[:-len(sfx)] + '_cleaned' + sfx
```

---

## YAML config for runtime options

Load all tunable parameters from an external file so the script itself stays static:

```python
opts = yaml.safe_load(open(cfgFile))
```

Pass `cfgFile` as an argument to the top-level function; don't hardcode paths inside.
