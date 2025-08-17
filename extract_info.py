import re
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import logging

from utils.gemini_service import get_gemini_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# If true, ai will be called to clean up the extracted text
ENABLE_CLEAN = False

def get_nickname_dict():
    return {
        'BILL': 'WILLIAM', 'WILLIAM': 'BILL',
        'GREG': 'GREGORY', 'GREGORY': 'GREG',
        'BOB': 'ROBERT', 'ROBERT': 'BOB',
        'DAVE': 'DAVID', 'DAVID': 'DAVE',
        'MIKE': 'MICHAEL', 'MICHAEL': 'MIKE',
        'JIM': 'JAMES', 'JAMES': 'JIM',
        'TOM': 'THOMAS', 'THOMAS': 'TOM',
        'RICK': 'RICHARD', 'RICHARD': 'RICK',
        'DICK': 'RICHARD', 'RICHARD': 'DICK',
        'STEVE': 'STEPHEN', 'STEPHEN': 'STEVE',
        'CHRIS': 'CHRISTOPHER', 'CHRISTOPHER': 'CHRIS',
        'DAN': 'DANIEL', 'DANIEL': 'DAN',
        'MATT': 'MATTHEW', 'MATTHEW': 'MATT',
        'KEN': 'KENNETH', 'KENNETH': 'KEN',
        'DREW': 'ANDREW', 'ANDREW': 'DREW'
    }

def similarity(a, b):
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()

def match_names(bio_names, names_list, nickname_dict):
    matches = {}
    for bio_name in bio_names:
        bio_parts = bio_name.split()
        for name in names_list:
            name_parts = name.split()
            
            # Direct match
            if any(part in bio_parts for part in name_parts):
                matches[name] = bio_name
                break
                
            # Nickname match
            for name_part in name_parts:
                if name_part in nickname_dict:
                    alt_name = nickname_dict[name_part]
                    if alt_name in bio_parts:
                        matches[name] = bio_name
                        break
    
    logger.info(f"Matched {len(matches)} names from {len(names_list)} total names")
    return matches

def find_fuzzy_phrase_match(text, target_phrase, threshold=0.8):
    """
    Find a fuzzy match for a multi-word phrase in text.
    Returns the matched phrase if found, None otherwise.
    """
    target_words = target_phrase.upper().split()
    words = text.upper().split()
    
    # For two-word phrases, check consecutive words
    if len(target_words) == 2:
        for i in range(len(words) - 1):
            # Check if first letters match at minimum
            if (words[i][0] == target_words[0][0] and 
                words[i+1][0] == target_words[1][0]):
                # Check similarity of both words
                sim1 = similarity(words[i], target_words[0])
                sim2 = similarity(words[i+1], target_words[1])
                # Both words should meet threshold
                if sim1 >= threshold and sim2 >= threshold:
                    # Return the actual text (not uppercase)
                    actual_words = text.split()
                    for j in range(len(actual_words) - 1):
                        if (actual_words[j].upper() == words[i] and 
                            actual_words[j+1].upper() == words[i+1]):
                            return f"{actual_words[j]} {actual_words[j+1]}"
    
    # Fallback for single word (backward compatibility)
    elif len(target_words) == 1:
        for word in text.split():
            if similarity(word.upper(), target_words[0]) >= threshold:
                return word
    
    return None

def remove_page_markers(text):
    """
    Remove page markers like === Page {*} === and surrounding lines.
    Removes the line before, the marker line, and the two lines after.
    """
    if not text:
        return text
    
    lines = text.split('\n')
    # Pattern to match page markers like === Page 1 ===, === Page 23 ===, etc.
    page_pattern = re.compile(r'^===\s*Page\s*\d+\s*===$')
    
    # Find indices of lines with page markers
    marker_indices = []
    for i, line in enumerate(lines):
        if page_pattern.match(line.strip()):
            marker_indices.append(i)
    
    # Create a set of line indices to remove
    # For each marker at index n, remove n-1, n, n+1, n+2
    lines_to_remove = set()
    for idx in marker_indices:
        # Remove previous line (n-1)
        if idx > 0:
            lines_to_remove.add(idx - 1)
        # Remove marker line (n)
        lines_to_remove.add(idx)
        # Remove next two lines (n+1, n+2)
        if idx + 1 < len(lines):
            lines_to_remove.add(idx + 1)
        if idx + 2 < len(lines):
            lines_to_remove.add(idx + 2)
    
    # Keep only lines not in the removal set
    cleaned_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    
    return '\n'.join(cleaned_lines)

def extract_bio_info(bio_content, bio_name):
    # Find the person's section
    pattern = rf'\b{re.escape(bio_name)}\b.*?(?=\n[A-Z]+,|\Z)'
    match = re.search(pattern, bio_content, re.DOTALL)
    if not match:
        return '', ''
    
    person_section = match.group()
    
    # Find "Political Career" section (looking for two-word phrase)
    political_match = find_fuzzy_phrase_match(person_section, 'Political Career', threshold=0.7)
    if not political_match:
        # Try with just "Political" as fallback
        political_match = find_fuzzy_phrase_match(person_section, 'Political', threshold=0.8)
        if not political_match:
            return '', ''
    
    political_start = person_section.find(political_match)
    after_political = person_section[political_start:]
    
    # Find "Private Career" section (looking for two-word phrase)
    private_match = find_fuzzy_phrase_match(after_political, 'Private Career', threshold=0.7)
    if not private_match:
        # Try with just "Private" as fallback
        private_match = find_fuzzy_phrase_match(after_political, 'Private', threshold=0.8)
        if not private_match:
            # No private career section found, return just political content
            # Remove the header text (e.g., "Political Career:" or variations)
            political_content = after_political
            # Try to clean up the header
            for possible_header in [political_match + ':', political_match]:
                if political_content.startswith(possible_header):
                    political_content = political_content[len(possible_header):].strip()
                    break
            # Remove page markers from political content
            political_content = remove_page_markers(political_content)
            return political_content, ''
    
    private_start = after_political.find(private_match)
    political_content = after_political[:private_start]
    
    # Clean up political content header
    for possible_header in [political_match + ':', political_match]:
        if political_content.startswith(possible_header):
            political_content = political_content[len(possible_header):].strip()
            break
    
    # Find Address section or end of private section
    after_private = after_political[private_start:]
    address_match = find_fuzzy_phrase_match(after_private, 'Address', threshold=0.8)
    
    if not address_match:
        # No address section, take everything after private header
        private_content = after_private
    else:
        address_start = after_private.find(address_match)
        private_content = after_private[:address_start]
    
    # Clean up private content header
    for possible_header in [private_match + ':', private_match]:
        if private_content.startswith(possible_header):
            private_content = private_content[len(possible_header):].strip()
            break
    
    # Remove page markers from both political and private content
    political_content = remove_page_markers(political_content.strip())
    private_content = remove_page_markers(private_content.strip())
    
    return political_content, private_content

def extract_names_from_file(file_path):
    logger.info(f"Extracting names from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    year_pattern = r'\b\d{4}\b'
    year_matches = list(re.finditer(year_pattern, content))
    if len(year_matches) >= 2:
        content = content[year_matches[1].end():]
    
    pattern = r'\b(?:HON[.,:]*\s+)?([A-Z][A-Z\s\.-]+?)(?=\s+(?:Minister|Premier|Deputy|\n))'
    matches = re.findall(pattern, content)
    names = [' '.join(match.strip().split()).rstrip('.') for match in matches if match.strip()]
    
    logger.info(f"Extracted {len(names)} names from {file_path.name}")
    return names

def process_names_files():
    txt_data_dir = Path('txt_data')
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    nickname_dict = get_nickname_dict()
    
    clean_prompt_template_path = Path("prompts/clean_response.txt")
    if not clean_prompt_template_path.exists():
        logger.error(f"Prompt template not found at {clean_prompt_template_path}. Please create it.")
        return
    clean_prompt_base = clean_prompt_template_path.read_text()
    
    logger.info(f"Starting processing. Looking for *Names.txt files in {txt_data_dir}")
    
    names_files = list(txt_data_dir.glob('*Names.txt'))
    logger.info(f"Found {len(names_files)} Names.txt files to process")
    
    for names_file in names_files:
        logger.info(f"Processing {names_file.name}")
        
        filename_parts = names_file.stem.split('_')
        if len(filename_parts) >= 3 and filename_parts[-1] == 'Names':
            year = filename_parts[0]
            base_name = '_'.join(filename_parts[1:-1])
            csv_filename = f"{year}_{base_name}.csv"
            bio_filename = f"{year}_{base_name}_Bio.txt"
            bio_path = txt_data_dir / bio_filename

            names = extract_names_from_file(names_file)
            if not names:
                logger.warning(f"No names found in {names_file.name}, skipping")
                continue
            
            first_names = []
            last_names = []
            
            for name in names:
                name_parts = name.strip().split()
                if len(name_parts) >= 2:
                    first_names.append(name_parts[0])
                    last_names.append(name_parts[-1]) 
                elif len(name_parts) == 1:
                    first_names.append(name_parts[0])
                    last_names.append('')
                else:
                    first_names.append('')
                    last_names.append('')
                    
            # Initialize with empty positions
            political_positions = [''] * len(names)
            private_positions = [''] * len(names)
            
            # Process bio file if it exists
            if bio_path.exists():
                logger.info(f"Processing bio file: {bio_filename}")
                with open(bio_path, 'r', encoding='utf-8') as bio_file:
                    bio_content = bio_file.read()
                
                # Extract all capitalized names from bio (remove HON prefix)
                bio_pattern = r'(?:HON\s+)?([A-Z][A-Z\s,.-]+?)(?=\s|,|\n)'
                bio_matches = re.findall(bio_pattern, bio_content)
                bio_names = [' '.join(match.strip().split()).rstrip('.,') for match in bio_matches if len(match.strip()) > 2]
                
                # Match names from names file to bio names
                name_matches = match_names(bio_names, names, nickname_dict)
                
                # Extract career info for matched names
                processed_count = 0
                for i, name in enumerate(names):
                    if name in name_matches:
                        political, private = extract_bio_info(bio_content, name_matches[name])
                        
                        if ENABLE_CLEAN:
                            if political:
                                logger.info(f"Cleaning political career for {name}...")
                                full_prompt = clean_prompt_base + political
                                political = get_gemini_response(full_prompt)

                            if private:
                                logger.info(f"Cleaning private career for {name}...")
                                full_prompt = clean_prompt_base + private
                                private = get_gemini_response(full_prompt)
                            
                        political_positions[i] = political
                        private_positions[i] = private
                        processed_count += 1
                
                logger.info(f"Extracted career info for {processed_count} matched names")
            else:
                logger.warning(f"Bio file {bio_filename} not found, creating CSV with names only")
            
            df = pd.DataFrame({
                'First Name': first_names,
                'Last Name': last_names,
                'Political Career': political_positions,
                'Private Career': private_positions
            })
            
            output_path = result_dir / csv_filename
            df.to_csv(output_path, index=False)
            logger.info(f"Created {csv_filename} with {len(names)} entries")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    process_names_files()