# LinkedIn post templates

## Option 1: technical and transparent

I have open-sourced an educational XVA simulation lab in Python and PyTorch.

It is built to make a few important quantitative-finance ideas inspectable:

- risk-neutral spot, variance, and short-rate simulation;
- pathwise exposure, collateral, MPOR, and first-to-default CVA/DVA;
- clearly labelled funding, IM, and capital proxies;
- a small Deep-BSDE experiment driven by the simulation's actual Brownian increments;
- static option-surface repair in call-price space with explicit strike and expiry constraints.

The repository is intentionally transparent about what it is not: it is not a
production pricing system, a regulatory model, a calibrated market-data stack,
or a proof of no-arbitrage.

That distinction matters. A useful quant project should make its assumptions,
approximations, and tests visible instead of overstating its scope.

Repository:
https://github.com/AIM-IT4/Thermodynamic_BSDE_XVA_Engine

Suggested attachments:
- xva_thermodynamic_diagnostics.png
- xva_3d_surfaces.png

#QuantFinance #XVA #DerivativesPricing #Python #PyTorch #MonteCarlo

---

## Option 2: educational

How do collateral, funding, default risk, and market paths interact in an XVA
calculation?

I published a compact teaching lab that walks through the mechanics with
reproducible code:

- exposure and PFE profiles;
- VM thresholds, MTA, call frequency, and MPOR;
- pathwise first-to-default CVA and DVA;
- simplified, clearly labelled MVA and KVA proxies;
- an optional neural BSDE experiment;
- a static option-surface repair exercise.

The code includes tests for numerical invariants and benchmark cases, along
with explicit limitations. It is designed for learning and review, not for
live trading or regulatory use.

https://github.com/AIM-IT4/Thermodynamic_BSDE_XVA_Engine

#QuantitativeFinance #XVA #FinancialEngineering #Python #RiskManagement
