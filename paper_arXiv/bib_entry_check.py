import os
import sys
import bibtexparser

def filter_article_fields(entry):
    """Keep only author, journal, title, and year fields in article entries."""
    allowed_fields = {'author', 'journal', 'title', 'year', 'ENTRYTYPE', 'ID'}
    fields_to_remove = set(entry.keys()) - allowed_fields
    for field in fields_to_remove:
        del entry[field]

def filter_inproceedings_fields(entry):
    """Keep only author, booktitle, publisher, title, and year fields in inproceedings entries."""
    allowed_fields = {'author', 'booktitle', 'publisher', 'title', 'year', 'ENTRYTYPE', 'ID'}
    fields_to_remove = set(entry.keys()) - allowed_fields
    for field in fields_to_remove:
        del entry[field]

def process_bib_file(file_path):
    """Process a .bib file to filter fields of article and inproceedings entries."""
    with open(file_path, 'r') as bib_file:
        bib_database = bibtexparser.load(bib_file)
    
    for entry in bib_database.entries:
        entry_type = entry.get('ENTRYTYPE', '').lower()
        if entry_type == 'article':
            filter_article_fields(entry)
        elif entry_type == 'inproceedings':
            filter_inproceedings_fields(entry)
    
    with open(file_path, 'w') as bib_file:
        writer = bibtexparser.bwriter.BibTexWriter()
        bib_file.write(writer.write(bib_database))

def main(file_path):
    """Main function to process the specified .bib file."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    if not file_path.endswith('.bib'):
        print(f"Error: File '{file_path}' is not a .bib file.")
        return
    
    print(f"Processing {file_path}...")
    process_bib_file(file_path)
    print("Processing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_bib_fields.py <path_to_bib_file>")
    else:
        main(sys.argv[1])