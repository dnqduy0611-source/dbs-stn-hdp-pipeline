# De-identification gate for DBS imaging archives

`phi_scrub.py` pseudonymises a batch of derived imaging files (MRtrix `.tck`/`.mif`,
dcm2niix `.json` sidecars, text logs, Slicer scenes) before they leave the local
machine, then runs a gate that must report **0 hits** before upload.

```
PHI_SCRUB_CONFIG=my_batch.json python phi_scrub.py scan|map|scrub|stage|rename|rescrub|manifest|gate
```

All identities (folder names, record numbers, institution names) live in an
external JSON config (see `phi_scrub_config.example.json`) that is **never committed**.

Design points learned the hard way:

1. **Binary headers are rewritten with same-length strings.** `.tck`/`.mif` headers
   contain a `file: . <offset>` pointer to the binary payload, so a name is replaced
   by `sub-NN` padded with `_` to the exact byte length. No offset changes; verified
   by reading every streamline back (`tckstats`), not just the header.
2. **One-pass gates miss name fragments.** Searching for full names misses folder
   names the project itself created (e.g. `bundles_<nickname>_10m`). The gate has a
   second pass over short fragments in headers and text files, with a whitelist of
   common code identifiers.
3. **Short IDs cause false alarms.** A 5-digit study ID matched random digits in
   ASCII `.vtk` coordinates, so the gate could never pass; such IDs are removed by
   JSON key instead of by string search.
4. **Documentation can leak too.** A restore note listing old file names failed the
   gate twice. The gate is always run last, after every file (including docs) is in
   its final place.
