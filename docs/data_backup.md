# Data backup policy

GitHub is used only for de-identified source code, templates, and
documentation. Patient data and generated scientific outputs must be stored
outside Git.

## Recommended layout

```text
encrypted-backup/
  patient-id-map/        Restricted identity mapping, stored separately
  source-data/           Immutable DICOM/NIfTI source data
  derivatives/           MRtrix, Lead-DBS, OSS-DBS and Slicer outputs
  manifests/             File list, size, SHA-256 checksum and backup date
```

Use pseudonymous subject IDs such as `sub-0001`. Do not place names, hospital
IDs, dates of birth, accession numbers, or identifying DICOM metadata in
filenames, configuration, logs, screenshots, or manifests.

## Backup controls

- Encrypt data at rest and in transit.
- Restrict access to authorized project members.
- Keep at least two independent copies in different locations.
- Generate and verify SHA-256 checksums after each backup.
- Test restoration periodically.
- Document retention, consent, and institutional requirements.

Git LFS does not solve confidentiality, retention, or large-scale scientific
data management. It should not be used for this project's patient dataset.
