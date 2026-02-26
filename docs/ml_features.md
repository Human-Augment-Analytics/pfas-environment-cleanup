# PFAS Adsorption Modeling – Feature Rationale
Sources:

https://pubs.acs.org/doi/pdf/10.1021/acsestwater.3c00569?ref=article_openPDF

https://pdf.sciencedirectassets.com/271852/1-s2.0-S0045653525X00072/1-s2.0-S0045653525002656/main.pdf

https://pubs.acs.org/doi/pdf/10.1021/acsenvironau.5c00081?ref=article_openPDF

https://pubs.acs.org/doi/pdf/10.1021/acsmaterialsau.3c00066?ref=article_openPDF

## Overview

This document outlines the compound features used in the small machine learning models for PFAS adsorption prediction and provides justification for their inclusion.

The selected features are based on known chemical interaction mechanisms governing PFAS binding, including:

- Electrostatic interactions
- Hydrogen bonding and polar interactions
- Hydrophobic and fluorophilic interactions
- Steric accessibility and molecular size effects

These features are designed to be chemically interpretable and directly aligned with PFAS adsorption mechanisms described in the literature.

---

# 1. Electrostatic Features

### Features Used
- `Charge`
- `flag_quat_ammonium`
- `flag_imidazolium`
- `flag_pyridinium`
- `flag_guanidine`

### Rationale

Most PFAS compounds exist as anions in aqueous environments (carboxylates or sulfonates). Therefore, positively charged adsorbent sites promote strong electrostatic attraction and ion-exchange behavior.

Permanent cationic functional groups such as quaternary ammonium and imidazolium structures are widely used in PFAS removal technologies due to their ability to bind anionic species.

Including these features allows the model to capture one of the dominant PFAS adsorption mechanisms: anion exchange.

---

# 2. Hydrogen Bonding and Polar Interaction Features

### Features Used
- `HBondDonorCount`
- `HBondAcceptorCount`
- `TPSA` (Topological Polar Surface Area)
- `flag_urea`
- `flag_thiourea`
- `flag_sulfonamide`

### Rationale

PFAS headgroups (–COO⁻ and –SO₃⁻) can participate in hydrogen bonding and polar interactions.

Structured hydrogen bonding motifs such as urea and thiourea provide strong directional interactions and may stabilize charged PFAS headgroups.

General hydrogen bond counts and polar surface area reflect the molecule’s ability to engage in polar interactions and influence solvation effects in water.

These features model secondary interaction mechanisms that complement electrostatic attraction.

---

# 3. Hydrophobic and Fluorophilic Features

### Features Used
- `XLogP`
- `flag_aromatic`
- `flag_fluorinated`

### Rationale

PFAS molecules contain highly fluorinated hydrophobic tails. Adsorbents with hydrophobic domains can enhance adsorption through nonpolar interactions.

- `XLogP` estimates overall hydrophobic character.
- Aromatic systems may enable π-type interactions.
- Fluorinated motifs may promote fluorophilic interactions ("like-with-like" effects).

These features allow the model to capture tail–tail interactions that are important for long-chain PFAS adsorption.

---

# 4. Molecular Size and Flexibility Features

### Features Used
- `MolecularWeight`
- `ExactMass`
- `RotatableBondCount`

### Rationale

Adsorption efficiency is influenced by steric accessibility and molecular flexibility.

- Larger molecules may experience steric hindrance.
- Excessive molecular weight can reduce diffusion and accessibility.
- Rotatable bond count reflects conformational flexibility, which affects interaction geometry.

These features help model structural constraints that may limit adsorption despite favorable chemistry.

---

# 5. PFAS Identity / Classification Features

### Features Used
- `pfas_PFOA`, `pfas_PFOS`, `pfas_PFBA`, `pfas_PFBS`, `pfas_PFPrA`, `pfas_HFPO-DA`, `pfas_TFSI`, `pfas_TFA`
- `pfas_is_long`
- `pfas_is_short`
- `pfas_is_ultrashort`

### Rationale

Adsorption behavior varies significantly across PFAS compounds due to differences in:

- Chain length
- Headgroup chemistry
- Hydrophobicity

Identity flags allow the model to distinguish adsorption behavior across PFAS types.

Classification features (long, short, ultrashort) encode structural trends that influence hydrophobic interactions and binding strength.

---

**Author:** Arjot Rai  
