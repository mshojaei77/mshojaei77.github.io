# 🛠️ Book Automation Scripts

This directory contains automation tools for maintaining "LLMs: From Foundation to Production".

## 📜 Available Scripts

### `update-progress.py`
**Purpose:** Automatically scans all book chapters and updates progress tracking in `book/index.md`

**Usage:**
```bash
# Run from project root
python scripts/update-progress.py
```

**What it does:**
- Scans all chapter files in `book/part*` directories
- Analyzes content to determine completion status
- Updates progress bar and chapter tables in book index
- Prints summary of all chapter statuses

**Status Detection Logic:**
- **✅ Published**: >3000 words, 5+ sections, 2+ code blocks, has summary
- **🚧 In Progress**: >1500 words, 3+ sections
- **📝 Draft**: >500 words
- **📋 Planned**: Everything else

## 🔧 Setup

**Install dependencies:**
```bash
uv pip install -r requirements-dev.txt
```

**Development workflow:**
1. Write/update chapter content
2. Run progress tracker: `python scripts/update-progress.py`
3. Commit changes: `git add . && git commit -m "[Book] Update progress tracking"`
4. Push to GitHub Pages

## 📋 Planned Scripts

### `validate-chapter.py` (TODO)
- Check chapter formatting consistency
- Validate front matter structure
- Verify all links work
- Check code syntax

### `test-code-examples.py` (TODO)
- Extract and run all Python code blocks
- Verify imports and dependencies
- Test example outputs
- Generate test reports

### `generate-toc.py` (TODO)
- Auto-generate table of contents
- Update cross-references
- Maintain navigation consistency

### `export-pdf.py` (TODO)
- Generate PDF version of complete book
- Combine all chapters into single document
- Maintain formatting and links

## 🎯 Best Practices

**Before running scripts:**
- Ensure you're in the project root directory
- Have latest changes committed to git
- Back up important files

**After running scripts:**
- Review changes before committing
- Test Jekyll site builds locally
- Verify GitHub Pages deployment

## 🐛 Troubleshooting

**Common Issues:**
- **"No module named 'pathlib'"**: Install requirements with `uv pip install -r requirements-dev.txt`
- **Permission errors**: Run from project root with appropriate permissions
- **Regex not matching**: Check chapter filename format (`XX_chapter_name.md`)

**Getting Help:**
- Check script output for error messages
- Verify file structure matches expected format
- Report issues in GitHub repository 