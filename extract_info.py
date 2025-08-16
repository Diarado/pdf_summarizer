import re
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

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

def find_fuzzy_match(text, target_word, threshold=0.8):
    words = text.split()
    for word in words:
        if similarity(word, target_word) >= threshold:
            return word
    return None

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
    return matches

def extract_bio_info(bio_content, bio_name):
    # Find the person's section
    pattern = rf'\b{re.escape(bio_name)}\b.*?(?=\n[A-Z]+,|\Z)'
    match = re.search(pattern, bio_content, re.DOTALL)
    if not match:
        return '', ''
    
    person_section = match.group()
    
    # Find Political Career section
    political_match = find_fuzzy_match(person_section, 'Political')
    if not political_match:
        return '', ''
    
    political_start = person_section.find(political_match)
    after_political = person_section[political_start:]
    
    # Find Private Career section
    private_match = find_fuzzy_match(after_political, 'Private')
    if not private_match:
        return after_political.replace(political_match + ' Career:', '').strip(), ''
    
    private_start = after_political.find(private_match)
    political_content = after_political[:private_start].replace(political_match + ' Career:', '').strip()
    
    # Find Address section
    after_private = after_political[private_start:]
    address_match = find_fuzzy_match(after_private, 'Address')
    if not address_match:
        private_content = after_private.replace(private_match + ' Career:', '').strip()
    else:
        address_start = after_private.find(address_match)
        private_content = after_private[:address_start].replace(private_match + ' Career:', '').strip()
    
    return political_content, private_content

def extract_names_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    year_pattern = r'\b\d{4}\b'
    year_matches = list(re.finditer(year_pattern, content))
    if len(year_matches) >= 2:
        content = content[year_matches[1].end():]
    
    pattern = r'\b(?:HON[.,:]*\s+)?([A-Z][A-Z\s\.-]+?)(?=\s+(?:Minister|Premier|Deputy|\n))'
    matches = re.findall(pattern, content)
    return [' '.join(match.strip().split()).rstrip('.') for match in matches if match.strip()]

def process_names_files():
    txt_data_dir = Path('txt_data')
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    nickname_dict = get_nickname_dict()
    
    for names_file in txt_data_dir.glob('*Names.txt'):
        filename_parts = names_file.stem.split('_')
        if len(filename_parts) >= 3 and filename_parts[-1] == 'Names':
            year = filename_parts[0]
            base_name = '_'.join(filename_parts[1:-1])
            csv_filename = f"{year}_{base_name}.csv"
            bio_filename = f"{year}_{base_name}_Bio.txt"
            bio_path = txt_data_dir / bio_filename
            
            names = extract_names_from_file(names_file)
            if not names:
                continue
                
            # Initialize with empty positions
            political_positions = [''] * len(names)
            private_positions = [''] * len(names)
            
            # Process bio file if it exists
            if bio_path.exists():
                with open(bio_path, 'r', encoding='utf-8') as bio_file:
                    bio_content = bio_file.read()
                
                # Extract all capitalized names from bio (remove HON prefix)
                bio_pattern = r'(?:HON\s+)?([A-Z][A-Z\s,.-]+?)(?=\s|,|\n)'
                bio_matches = re.findall(bio_pattern, bio_content)
                bio_names = [' '.join(match.strip().split()).rstrip('.,') for match in bio_matches if len(match.strip()) > 2]
                
                # Match names from names file to bio names
                name_matches = match_names(bio_names, names, nickname_dict)
                
                # Extract career info for matched names
                for i, name in enumerate(names):
                    if name in name_matches:
                        political, private = extract_bio_info(bio_content, name_matches[name])
                        political_positions[i] = political
                        private_positions[i] = private
            
            df = pd.DataFrame({
                'Name': names,
                'Political Career': political_positions,
                'Private Career': private_positions
            })
            df.to_csv(result_dir / csv_filename, index=False)

if __name__ == "__main__":
    process_names_files()