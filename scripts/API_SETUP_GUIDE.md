# API Setup Guide - Mega Fruit Scraper

This guide will help you obtain FREE API keys for 3 professional photography sources.
**Total setup time: ~10 minutes**

---

## 1. Unsplash API (Recommended - Highest Quality)

### Registration Steps:
1. Go to: https://unsplash.com/join
2. Create a free account (email + password)
3. Go to: https://unsplash.com/oauth/applications
4. Click **"New Application"**
5. Accept terms and conditions
6. Fill in:
   - **Application name**: "Fruit Classification ML Project"
   - **Description**: "Educational machine learning project for fruit image classification"
7. Click **"Create application"**
8. Copy your **Access Key** (starts with something like `abc123...`)

### Rate Limits:
- **50 requests per hour** (enough for ~1,500 images/hour)
- Demo/Development tier is FREE forever

---

## 2. Pexels API (Recommended - Great Variety)

### Registration Steps:
1. Go to: https://www.pexels.com/
2. Create a free account
3. Go to: https://www.pexels.com/api/new/
4. Fill in:
   - **Project name**: "Fruit Classification Project"
   - **Project description**: "Educational ML project for fruit recognition"
   - **Project URL**: (optional, can leave blank or put GitHub)
5. Click **"Generate API Key"**
6. Copy your **API Key** (starts with something like `563492ad6f91...`)

### Rate Limits:
- **200 requests per hour**
- **20,000 requests per month**
- FREE forever

---

## 3. Pixabay API (Optional - Most Images)

### Registration Steps:
1. Go to: https://pixabay.com/
2. Create a free account
3. Go to: https://pixabay.com/api/docs/#api_register
4. Click **"Get your API key"**
5. Your API key will be displayed immediately (a long string)
6. Copy it

### Rate Limits:
- **100 requests per minute**
- **5,000 requests per hour**
- FREE tier: perfect for our needs

---

## Quick Start Commands

### After getting your API keys:

```bash
# Navigate to project
cd /home/jaime/AF

# Run the scraper
python scripts/mega_fruit_scraper.py
```

### You'll be prompted to enter:
1. Unsplash Access Key
2. Pexels API Key
3. Pixabay API Key
4. Images per fruit (recommended: 300)

---

## Expected Results

With **300 images per fruit** target:

| Source | Images/Fruit | Total |
|--------|-------------|-------|
| Unsplash | ~120 (40%) | ~2,640 |
| Pexels | ~90 (30%) | ~1,980 |
| Pixabay | ~90 (30%) | ~1,980 |
| **TOTAL** | **~300** | **~6,600** |

---

## Troubleshooting

### "Rate limit exceeded"
- Wait 1 hour and resume
- Script automatically handles this

### "No results found"
- Some rare fruits (chirimoya, paraguayo) may have fewer images
- Script will download what's available

### "Connection timeout"
- Check your internet connection
- Script will retry automatically

---

## Legal Notice

All images from these sources are:
- ✅ **Free to use** for educational/research purposes
- ✅ **High quality** professional photography
- ✅ **Legally obtained** through official APIs
- ✅ **No copyright issues**

Perfect for academic ML projects!

---

## Alternative: Pre-configured Script

If you want to skip API registration, you can use **Bing Image Search** (no API key needed but lower quality):

```bash
pip install bing-image-downloader
python scripts/simple_bing_scraper.py
```

But I **strongly recommend** using the main scraper with APIs for better quality and reliability.
