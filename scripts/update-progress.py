#!/usr/bin/env python3
"""
Book Progress Tracker
Automatically updates progress tracking in book/index.md based on chapter completion status
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def scan_chapters() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Scan all book parts and chapters to determine completion status
    
    Returns:
        Dict mapping part names to list of (chapter_num, title, status) tuples
    """
    book_dir = Path("book")
    parts = {}
    
    for part_dir in sorted(book_dir.glob("part*")):
        if not part_dir.is_dir():
            continue
            
        part_name = part_dir.name
        chapters = []
        
        for chapter_file in sorted(part_dir.glob("*.md")):
            if chapter_file.name == "index.md":
                continue
                
            # Extract chapter number and title from filename
            match = re.match(r"(\d+)_(.+)\.md", chapter_file.name)
            if not match:
                continue
                
            chapter_num = match.group(1)
            title_slug = match.group(2)
            
            # Read file to get actual title and determine status
            with open(chapter_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract title from front matter
            title_match = re.search(r'title:\s*"Chapter \d+:\s*([^"]+)"', content)
            title = title_match.group(1) if title_match else title_slug.replace("_", " ").title()
            
            # Determine status based on content length and completeness
            status = determine_chapter_status(content)
            
            chapters.append((chapter_num, title, status))
            
        parts[part_name] = chapters
    
    return parts


def determine_chapter_status(content: str) -> str:
    """
    Determine chapter completion status based on content analysis
    
    Args:
        content: Full chapter content
        
    Returns:
        Status string: "✅ Published", "🚧 In Progress", "📝 Draft", "📋 Planned"
    """
    # Count sections, code blocks, and overall length
    sections = len(re.findall(r'^## ', content, re.MULTILINE))
    code_blocks = len(re.findall(r'```python', content))
    word_count = len(content.split())
    
    # Check for completion indicators
    has_exercises = "## 🧪 Exercises" in content
    has_summary = "## 💡 Summary" in content
    has_further_reading = "## 📚 Further Reading" in content
    
    if word_count > 3000 and sections >= 5 and code_blocks >= 2 and has_summary:
        return "✅ Published"
    elif word_count > 1500 and sections >= 3:
        return "🚧 In Progress"
    elif word_count > 500:
        return "📝 Draft"
    else:
        return "📋 Planned"


def calculate_progress(parts: Dict[str, List[Tuple[str, str, str]]]) -> Tuple[int, int, int]:
    """
    Calculate overall book progress statistics
    
    Returns:
        Tuple of (published_count, total_count, percentage)
    """
    total_chapters = 0
    published_chapters = 0
    
    for part_chapters in parts.values():
        total_chapters += len(part_chapters)
        published_chapters += sum(1 for _, _, status in part_chapters if status == "✅ Published")
    
    percentage = int((published_chapters / total_chapters) * 100) if total_chapters > 0 else 0
    
    return published_chapters, total_chapters, percentage


def update_book_index(parts: Dict[str, List[Tuple[str, str, str]]], progress: Tuple[int, int, int]) -> None:
    """
    Update the book/index.md file with current progress and chapter statuses
    """
    index_file = Path("book/index.md")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    published, total, percentage = progress
    
    # Update progress bar
    progress_pattern = r'<div class="progress-fill" style="width: \d+%"></div>'
    content = re.sub(progress_pattern, f'<div class="progress-fill" style="width: {percentage}%"></div>', content)
    
    # Update progress text
    progress_text_pattern = r'<div class="progress-text">\d+% Complete \(\d+/\d+ chapters published\)</div>'
    content = re.sub(progress_text_pattern, 
                    f'<div class="progress-text">{percentage}% Complete ({published}/{total} chapters published)</div>', 
                    content)
    
    # Update chapter tables
    for part_name, chapters in parts.items():
        part_mapping = {
            "part1-foundations": "Part I: Foundations",
            "part2-building-and-training": "Part II: Building & Training Models", 
            "part3-advanced-topics": "Part III: Advanced Topics & Specialization",
            "part4-engineering-and-applications": "Part IV: Engineering & Applications"
        }
        
        part_title = part_mapping.get(part_name, part_name)
        
        # Find and update the table for this part
        table_pattern = rf'(### [🔍🧬⚙️🚀] \[{re.escape(part_title)}\].*?\n\n)(.*?)(\n\n###|\n---|\Z)'
        
        def update_table(match):
            header = match.group(1)
            footer = match.group(3)
            
            # Build new table
            table_lines = [
                "| Chapter | Topic | Status | Est. Pages |",
                "|---------|-------|--------|------------|"
            ]
            
            for chapter_num, title, status in chapters:
                # Estimate pages based on content
                pages = estimate_pages(status)
                chapter_link = f"{part_name}/{chapter_num:0>2}_{title.lower().replace(' ', '_').replace('&', 'and')}.html"
                table_lines.append(f"| {chapter_num} | [{title}]({chapter_link}) | {status} | {pages} |")
            
            return header + "\n".join(table_lines) + footer
        
        content = re.sub(table_pattern, update_table, content, flags=re.DOTALL)
    
    # Write updated content
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(content)


def estimate_pages(status: str) -> int:
    """Estimate page count based on chapter status"""
    base_pages = {
        "✅ Published": 25,
        "🚧 In Progress": 20, 
        "📝 Draft": 15,
        "📋 Planned": 25
    }
    return base_pages.get(status, 25)


def main():
    """Main function to update book progress"""
    print("📚 Scanning book chapters...")
    parts = scan_chapters()
    
    print("📊 Calculating progress...")
    progress = calculate_progress(parts)
    published, total, percentage = progress
    
    print(f"📈 Progress: {published}/{total} chapters ({percentage}%)")
    
    print("✍️ Updating book index...")
    update_book_index(parts, progress)
    
    print("✅ Book progress updated successfully!")
    
    # Print summary
    print("\n📋 Chapter Status Summary:")
    for part_name, chapters in parts.items():
        print(f"\n{part_name}:")
        for chapter_num, title, status in chapters:
            print(f"  {chapter_num:2}. {title:<30} {status}")


if __name__ == "__main__":
    main() 