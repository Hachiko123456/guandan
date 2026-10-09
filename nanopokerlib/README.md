# nanopokerlib Guandan runtime

This directory contains the M04 C++/pybind Guandan runtime. Generic batch/type
concepts are derived from the MIT-licensed `nanopokerlib` reference at commit
`acdcf199cf422e2b683cb22142b8e7cb33b14460`. Guandan rules are a new
implementation based on this repository's accepted Python oracle; no Doudizhu
rule implementation is copied or renamed.

Build in place:

```powershell
python -m pip install pybind11
python setup.py build_ext --inplace
```

Import explicitly as `nanopokerlib.guandan`. Missing extensions fail at import;
there is no silent Python fallback.
