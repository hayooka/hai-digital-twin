# Dataset — HAI + HAIEnd

This project uses two public ICS security datasets, merged into one.

---

## Download

### HAI Dataset
- **Source:** https://github.com/icsdataset/hai
- **Paper:** HAI: HIL-based Augmented ICS Security Dataset

### HAIEnd Dataset
- **Source:** https://github.com/icsdataset/haiend
- **Paper:** HAIEnd: an extended HAI dataset

---

## Setup Instructions

1. Download both datasets from the links above
2. Place the raw CSV files in `data/raw/` with this structure:

```
data/
└── raw/
    ├── hai/
    │   ├── hai-train1.csv
    │   ├── hai-train2.csv
    │   ├── hai-train3.csv
    │   ├── hai-train4.csv
    │   ├── hai-test1.csv
    │   └── hai-test2.csv
    └── haiend/
        ├── haiend-train1.csv
        ├── haiend-train2.csv
        ├── haiend-train3.csv
        ├── haiend-train4.csv
        ├── haiend-test1.csv
        └── haiend-test2.csv
```

3. Run the data loader to merge and prepare:
```bash
python utils/data_loader.py
```

---

## Dataset Details

| Property | Value |
|----------|-------|
| Total features after merge | 277 sensors |
| HAI features | 86 |
| HAIEnd features | 225 |
| Duplicate columns removed | 35 |
| Train files | train1, train2, train3, train4 (100% benign) |
| Test files | test1, test2 (benign + attack rows) |
| Attack label column | `attack` (0 = normal, 1 = attack) |

---

## Note
Data files are **not included** in this repository due to size.
All raw CSV files are ignored by `.gitignore`.
