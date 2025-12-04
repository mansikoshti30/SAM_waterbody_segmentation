# GitHub Repository Setup Guide

This guide will help you upload your SAM Water Body Segmentation project to GitHub.

## Prerequisites

- Git installed on your system ([Download Git](https://git-scm.com/downloads))
- GitHub account ([Sign up](https://github.com/signup))

## Step-by-Step Guide

### 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top-right corner
3. Select **"New repository"**
4. Fill in the repository details:
   - **Repository name**: `SAM_project` (or your preferred name)
   - **Description**: "Water body segmentation from satellite imagery using Meta's Segment Anything Model (SAM)"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### 2. Initialize Git in Your Local Project

Open a terminal in your project directory and run:

```bash
# Navigate to your project directory
cd d:\projects\SAM_project

# Initialize git repository
git init

# Add all files to staging
git add .

# Commit your files
git commit -m "Initial commit: SAM water body segmentation project"
```

### 3. Connect to GitHub and Push

Replace `yourusername` with your actual GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/SAM_project.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Verify Upload

1. Go to your repository URL: `https://github.com/yourusername/SAM_project`
2. You should see all your files uploaded

## Important: Files NOT to Upload

The `.gitignore` file already excludes these, but double-check:

- ✅ **DO NOT upload**: `sam_vit_b.pth` (model checkpoint ~375MB)
- ✅ **DO NOT upload**: Large dataset files in `data_set/`
- ✅ **DO NOT upload**: Output folders (`batch_outputs*/`)
- ✅ **DO upload**: Python scripts, documentation, configuration files

## Recommended: Add Topics and Description

On your GitHub repository page:

1. Click **"About"** (gear icon)
2. Add topics: `python`, `satellite-imagery`, `deep-learning`, `SAM`, `water-segmentation`, `remote-sensing`, `geospatial`
3. Add website/documentation link (if you have one)

## Recommended: Create Repository Sections

### Add Repository Sections

Create these files if you want more structure:

1. **docs/** folder for detailed documentation
2. **examples/** folder for example scripts
3. **tests/** folder for unit tests

## Using Git Commands

### Common Git Workflows

**Make changes and update repository:**
```bash
git add .
git commit -m "Description of changes"
git push
```

**Check status:**
```bash
git status
```

**View commit history:**
```bash
git log --oneline
```

**Create a new branch:**
```bash
git checkout -b feature-name
```

**Merge branch:**
```bash
git checkout main
git merge feature-name
git push
```

## Optional: Set Up GitHub Pages

To create a documentation website:

1. Go to repository **Settings** → **Pages**
2. Source: Deploy from a branch → `main` → `/docs` or `/` (root)
3. Save

## Optional: Add Badges to README

Add status badges at the top of your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GitHub stars](https://img.shields.io/github/stars/yourusername/SAM_project)
```

## Troubleshooting

### Large File Error

If you accidentally try to upload large files:

```bash
# Remove large files from staging
git rm --cached sam_vit_b.pth
git rm --cached -r data_set/

# Commit the removal
git commit -m "Remove large files"
git push
```

### Authentication Issues

GitHub now requires a Personal Access Token (PAT) instead of password:

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`
4. Copy the token
5. Use it as your password when pushing

Or use SSH instead of HTTPS:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Change remote URL to SSH
git remote set-url origin git@github.com:yourusername/SAM_project.git
```

## Next Steps After Upload

1. **Add a GitHub Action** - Automated testing (already included in `.github/workflows/`)
2. **Enable Discussions** - Community engagement
3. **Create Issues** - Track bugs and features
4. **Add Contributors** - Invite collaborators
5. **Release Tags** - Version your releases
6. **Star the repository** - Show your support!

## Sample Repository URLs

- Repository: `https://github.com/yourusername/SAM_project`
- Clone URL (HTTPS): `https://github.com/yourusername/SAM_project.git`
- Clone URL (SSH): `git@github.com:yourusername/SAM_project.git`

## Additional Resources

- [GitHub Docs](https://docs.github.com)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Markdown Guide](https://guides.github.com/features/mastering-markdown/)

---

**Congratulations!** Your project is now on GitHub! 🎉
