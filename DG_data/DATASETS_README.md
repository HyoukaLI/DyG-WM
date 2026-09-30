# Datasets

The processed archives used in the paper are already included in
`processed_data/` (one `.npz` per dataset), so no download is needed to
reproduce the results. This folder is only needed to rebuild them from the
raw data.

| Dataset | Type | Raw source | Raw files expected here |
| --- | --- | --- | --- |
| Wikipedia | bipartite | JODIE (Kumar et al., KDD'19) | `wikipedia/wikipedia.csv` |
| MOOC | bipartite | JODIE | `mooc/mooc.csv` |
| LastFM | bipartite | JODIE | `lastfm/lastfm.csv` |
| Enron | homogeneous | DyGLib / Poursafaei et al. (NeurIPS'22) | `enron/ml_enron{.csv,.npy,_node.npy}` |
| UCI | homogeneous | DyGLib | `uci/ml_uci{.csv,.npy,_node.npy}` |
| Can. Parl. | homogeneous | DyGLib | `CanParl/ml_CanParl{.csv,.npy,_node.npy}` |
| Contact | homogeneous | DyGLib | `Contacts/ml_Contacts{.csv,.npy,_node.npy}` |
| Flights | homogeneous | DyGLib | `Flights/ml_Flights{.csv,.npy,_node.npy}` |
| UN Trade | homogeneous | DyGLib | `UNtrade/ml_UNtrade{.csv,.npy,_node.npy}` |
| UN Vote | homogeneous | DyGLib | `UNvote/ml_UNvote{.csv,.npy,_node.npy}` |
| US Legis. | homogeneous | DyGLib | `USLegis/ml_USLegis{.csv,.npy,_node.npy}` |

- JODIE CSVs: <http://snap.stanford.edu/jodie/>
- DyGLib preprocessed data: <https://zenodo.org/record/7213796#.Y1cO6y8r30o>
  (run DyGLib's `preprocess_data.py` to obtain the `ml_*` files)

Rebuild every archive with

```bash
bash preprocess_data/preprocess_all_data.sh
```

Each event stream is cut into 50 snapshots of equal event count. Duplicate
events, exact timestamps and event order are kept as the per-snapshot link
queries, so the continuous-time baselines still read the original stream;
feature normalisation is fit on the chronological training prefix only.
