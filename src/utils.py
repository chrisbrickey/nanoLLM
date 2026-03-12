from pathlib import Path

def load_stories_from_file(
    file_path,
    max_stories = None
):
    """
    Efficiently load stories from a text file.
    Each story ends with <|endoftext|>.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading stories from {file_path}...")
    stories = []
    current_story = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '<|endoftext|>' in line:
                parts = line.split('<|endoftext|>')
                for part in parts[:-1]:
                    current_story.append(part)
                    story_text = ''.join(current_story).strip()
                    if story_text:
                        stories.append(story_text + '<|endoftext|>')
                        if max_stories and len(stories) >= max_stories:
                            break
                    current_story = []
                if parts[-1].strip():
                    current_story = [parts[-1]]
                else:
                    current_story = []
                if max_stories and len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)
        if current_story and (not max_stories or len(stories) < max_stories):
            story_text = ''.join(current_story).strip()
            if story_text:
                stories.append(story_text + '<|endoftext|>')

    print(f"Loaded {len(stories):,} stories")
    return stories
