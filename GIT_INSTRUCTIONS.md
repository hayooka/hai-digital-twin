# 📖 How to Join the Project on GitHub
### For first-time Git users

---

## Step 1 — Install Git
Download from: https://git-scm.com/downloads
Install with default settings.

---

## Step 2 — Install VS Code (recommended)
Download from: https://code.visualstudio.com
This is where you will write and run your code.

---

## Step 3 — Open Terminal
In VS Code: **Terminal → New Terminal**

---

## Step 4 — Set up your identity (one time only)
```bash
git config --global user.name  "Your Name"
git config --global user.email "your_email@example.com"
```

---

## Step 5 — Clone the project
```bash
cd Desktop
git clone https://github.com/hayooka/hai-digital-twin.git
cd hai-digital-twin
```
Now you have the full project on your computer.

---

## Step 6 — Create your own branch
**Never work on main directly!**
```bash
git checkout -b feature/your-name
```

Examples:
```bash
git checkout -b feature/transformer    # Person working on Transformer
git checkout -b feature/vae            # Person working on VAE
git checkout -b feature/diffusion      # Person working on Diffusion
git checkout -b feature/iso-forest     # Person working on ISO Forest
```

---

## Step 7 — Daily workflow

### When you start working:
```bash
# 1. Make sure you are on your branch
git checkout feature/your-name

# 2. Pull latest changes from main
git pull origin main
```

### After you write code:
```bash
# 1. See what changed
git status

# 2. Add your files
git add models/your_model.py

# 3. Save your work with a message
git commit -m "add transformer architecture"

# 4. Push to GitHub
git push origin feature/your-name
```

---

## Step 8 — Rules ⚠️

```
✅ Always work on YOUR branch
✅ Never push to main directly
✅ Commit often — small changes are better
✅ Write clear commit messages

❌ Never delete someone else's file
❌ Never git push origin main
❌ Never edit utils/ files without telling Farah
```

---

## File ownership — who owns what

| Person | Branch | Files |
|--------|--------|-------|
| Farah  | main | utils/, data pipeline |
| Person 2 | feature/lstm | models/lstm_seq2seq.py |
| Person 3 | feature/transformer | models/transformer.py |
| Person 4 | feature/vae + feature/diffusion | models/vae.py, models/diffusion.py |
| Person 5 | feature/iso-forest | models/iso_forest.py, twin/core/ |

---

## Common commands — cheat sheet

```bash
git status              # see what changed
git add .               # add all changes
git add filename.py     # add one file
git commit -m "message" # save changes
git push                # upload to GitHub
git pull origin main    # get latest from main
git branch              # see your current branch
git log --oneline       # see commit history
```

---

## If something goes wrong 😱

**Don't panic! Just tell Farah.**

Or run:
```bash
git status    # see what happened
git diff      # see what changed in files
```

---

## Need help?
Ask Farah or open an issue on GitHub:
https://github.com/hayooka/hai-digital-twin/issues
