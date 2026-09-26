# Tier 2.3 multi-repository comparison

The original Flask, Requests, and FastAPI artifacts remain unchanged. Two
additional same-domain Python repositories were acquired at pinned commits:
Django (`4fab678a0739d54401ccee7eb587553657c9f76e`) and aiohttp
(`1133ceb762e67f27d6b64f0c541e01c53a60760b`). Their datasets use the same
collector and preprocessing pipeline, with an explicit deterministic cap of
100 sorted Python files because full Django history traversal is not bounded
enough for this local experiment. This cap is part of the recorded protocol,
not an implicit timeout.

All listed experiments use `base_seed=42`, 20 comparison trials, 50 Optuna
tuning trials, population 15, and 10 GA generations. The generated JSON
artifacts include raw trial arrays, configuration, source commit, and timing.

| Repository | Clean rows | Features | GA selected | All-feature MSE | GA ANN MSE | Tuned GA-XGB MSE | GA time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flask | 80 | 17 | 7 | 0.391898 | 0.339299 | 0.492758 | 49.4 |
| Requests | 36 | 16 | 9 | 0.535466 | 0.397316 | 0.883636 | 21.6 |
| Django (100-file bounded sample) | 54 | 14 | 7 | 0.924691 | 0.936432 | 0.932231 | 13.1 |
| aiohttp (100-file bounded sample) | 100 | 18 | 6 | 1.560527 | 1.391042 | 2.200854 | 33.3 |

The expanded result is mixed: GA improves the all-feature mean for Flask,
Requests, and aiohttp, while the bounded Django sample is effectively tied
and slightly worse. This is the direct cross-repository result and should not
be generalized to full Django without a full-scale collection run.

Dataset SHA-256 values:

- Flask: `8170c01b07032dc70ba38e38fca1000d66f15693d0b81086e1b5283e30449dec`
- Requests: `6f725fbef00f1e51f739823ba6647c1638a00212ac335ac693475c8b1b60a519`
- Django: `fdc0d3f7e85d0575476d105d3cd96aef3fa15195154708e8e1119484a2920736`
- aiohttp: `9c3da6b6241727224d84349e029937fc4342491d7284e0fe164ed6436ce2e62c`
