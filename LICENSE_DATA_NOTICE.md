# Dataset Attribution, Erratum Notice, and License Terms

## Dataset Credit & Citation

This repository is designed to analyze the **corrected Criteo Uplift Prediction Dataset (`criteo-research-uplift-v2.1.csv.gz`)**, published by **Criteo AI Lab** alongside the paper:

> Eustache Diemert, Artem Betlei, Christophe Renaudin (Criteo AI Lab), and Massih-Reza Amini (LIG, Grenoble INP).
> *"A Large Scale Benchmark for Uplift Modeling."*
> Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August 20, 2018 (ACM).

```bibtex
@inproceedings{Diemert2018,
  author    = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
  title     = {A Large Scale Benchmark for Uplift Modeling},
  publisher = {ACM},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August, 20, 2018},
  year      = {2018}
}
```

## Official Sources & v2.1 Erratum

- **Publisher, Erratum & License Page:** https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- **Primary Direct Archive Link:** http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
- **Publisher-Maintained Mirror (Hugging Face):** https://huggingface.co/datasets/criteo/criteo-uplift/tree/main
  - Direct file URL: https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
- **Expected File Name:** `criteo-research-uplift-v2.1.csv.gz`
- **Expected Compressed Size:** `311,422,618` bytes
- **Expected SHA-256 Checksum:** `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- **Expected Full Row Count:** `13,979,592` rows across 16 comma-separated columns (`f0`–`f11`, `treatment`, `conversion`, `visit`, `exposure`).

### Why v2.1 Is Required (Publisher Erratum)
In the initial v1 release (`25,309,483` rows), non-uniformity of incrementality levels across pooled advertisers introduced an information leak: models could infer advertiser identity from anonymized feature distributions (`f0`–`f11`) and artificially inflate uplift performance. Criteo AI Lab released `criteo-research-uplift-v2.1.csv.gz` (`13,979,592` rows; average visit rate `0.046992`; average conversion rate `0.00292`; treatment ratio `0.85`) to remove that cross-advertiser leak.

## License Summary (CC BY-NC-SA 4.0)

Criteo AI Lab distributes the Criteo Uplift Prediction Dataset under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)**:
- **Attribution (BY):** Credit Criteo AI Lab and Diemert et al. (2018), link to the license and dataset URI, and indicate whether changes were made.
- **NonCommercial (NC):** Use of the dataset is restricted to non-commercial research, educational, and portfolio evaluation purposes—not commercial advantage or monetary compensation.
- **ShareAlike (SA):** Any shared adapted database material derived from the dataset must carry CC BY-NC-SA 4.0 or a compatible license.
- **No Raw Data in Repository:** Raw Criteo data is intentionally excluded from this repository (`.gitignore` blocks `data/raw/*.csv*`). The included `data/fixtures/synthetic_criteo_fixture.csv.gz` file is 100% synthetic data generated from random number generators for offline unit and pipeline testing only.
- **Full Legal Code:** https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
