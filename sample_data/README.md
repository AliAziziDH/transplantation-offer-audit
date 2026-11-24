# Synthetic Sample Data (`100% SYNTHETIC`)

**Every file in `sample_data/` is 100% synthetically generated** by
[`src/scripts/generate_synthetic_data.py`](../src/scripts/generate_synthetic_data.py)
using a seeded pseudo-random generator (`np.random.default_rng(42)`).

- **Zero real clinical, donor, candidate, recipient, hospital, or UNOS/OPTN data** is
  contained in this directory.
- `synthetic_ki_offers.csv` (`2,000` rows × `102` columns) and
  `synthetic_ki_outcomes.csv` (`640` rows × `68` columns) preserve the exact column
  names, data types, and structural quirks of the original offer/outcome extracts solely
  so that the database loader (`src/scripts/create_db.py`) and audit/evaluation script
  (`src/scripts/eval_transplantation.py`) can be executed end-to-end out of the box.
- **Do not treat metrics computed on `sample_data/` as real-world clinical accuracy.**
  Real-data audit measurements are documented separately in [`../AUDIT.md`](../AUDIT.md).
