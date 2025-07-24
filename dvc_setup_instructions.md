# DVC Google Drive Setup Instructions

## 🎯 What We've Accomplished

✅ **DVC Initialized**: Data Version Control is set up and working
✅ **Data Tracked**: California Housing dataset (1.9MB, 20,640 rows) is tracked with DVC
✅ **Git Integration**: DVC metadata files are tracked in git, actual data files are ignored
✅ **Auto-staging**: Enabled for seamless workflow

## 📁 DVC File Structure

```
data/raw/california_housing.csv      # ← Actual data (git-ignored)
data/raw/california_housing.csv.dvc  # ← DVC metadata (git-tracked)
```

The `.dvc` file contains:
- **MD5 hash**: `fa9fe4cf24f70b69ac65fb33062ddf34`
- **File size**: 1,915,795 bytes (1.9MB)
- **Path**: Reference to the actual data file

## 🔧 Optional: Google Drive Remote Setup

To enable team collaboration and cloud backup:

### 1. Create Google Drive Folder
1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder (e.g., "MLOps-Housing-Data")
3. Get the folder ID from the URL: `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`

### 2. Configure DVC Remote
```bash
# Add Google Drive remote
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID

# Example:
# dvc remote add -d gdrive gdrive://1BxH7gF2kL9mN3oP4qR5sT6uV7wX8yZ9

# Verify remote configuration
dvc remote list
```

### 3. Update Environment Variables
Add to your `.env` file:
```bash
GDRIVE_FOLDER_ID=YOUR_FOLDER_ID
DVC_GDRIVE_USE_SERVICE_ACCOUNT=false
```

### 4. Push Data to Remote
```bash
# First time authentication (opens browser)
dvc push

# This will:
# 1. Upload data/raw/california_housing.csv to Google Drive
# 2. Store it in your configured folder
# 3. Enable team members to sync the same data
```

### 5. Team Collaboration Workflow
```bash
# New team member setup:
git clone <repository-url>
dvc pull  # Downloads data from Google Drive

# After data changes:
dvc add data/raw/new_dataset.csv
git add data/raw/new_dataset.csv.dvc
git commit -m "Add new dataset"
dvc push  # Upload to Google Drive
git push  # Share with team
```

## 🚀 DVC Commands Reference

```bash
# Check status
dvc status

# List tracked files
dvc list . data/raw

# Show file info
dvc data status data/raw/california_housing.csv

# Remove from tracking
dvc remove data/raw/california_housing.csv.dvc

# Reproduce pipeline (when we add ML pipelines)
dvc repro

# Compare versions
dvc diff

# Show data lineage
dvc dag
```

## 🔄 Workflow Benefits

1. **Version Control**: Track large datasets without bloating git
2. **Reproducibility**: Exact data versions for each experiment
3. **Collaboration**: Team members get same data automatically
4. **Storage Efficiency**: Deduplicated storage across versions
5. **Pipeline Tracking**: Link data versions to model versions

## 📊 Current Dataset Info

- **Name**: California Housing Dataset
- **Size**: 1,915,795 bytes (1.9 MB)
- **Rows**: 20,640
- **Columns**: 9
- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
- **Target**: MedHouseVal (median house value)
- **DVC Hash**: fa9fe4cf24f70b69ac65fb33062ddf34

## ⚠️ Important Notes

- **Never commit large data files to git** - DVC handles this
- **Always commit .dvc files to git** - These contain metadata
- **Use `dvc pull` after `git pull`** - To sync data changes
- **Test DVC setup** before team collaboration

---

**Next Steps**: Ready for model training with MLflow experiment tracking! 🚀 