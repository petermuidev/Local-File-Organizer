import re
import os
import time
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from data_processing_common import sanitize_filename

def summarize_text_content(input_text, text_inference):
    prompt = f"Summarize the following text in 100 words or less:\n\n{input_text[:2000]}"
    return text_inference(prompt)

def process_single_text_file(args, text_inference, silent=False, log_file=None):
    """Process a single text file to generate metadata."""
    file_path, text = args
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(file_path)}", total=1.0)
        foldername, filename, description = generate_text_metadata(text, file_path, progress, task_id, text_inference)

    end_time = time.time()
    time_taken = end_time - start_time

    message = f"File: {file_path}\nTime taken: {time_taken:.2f} seconds\nDescription: {description}\nFolder name: {foldername}\nGenerated filename: {filename}\n"
    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    else:
        print(message)
    return {
        'file_path': file_path,
        'foldername': foldername,
        'filename': filename,
        'description': description
    }

def process_text_files(text_tuples, text_inference, silent=False, log_file=None):
    """Process text files sequentially."""
    results = []
    for args in text_tuples:
        data = process_single_text_file(args, text_inference, silent=silent, log_file=log_file)
        results.append(data)
    return results

def generate_text_metadata(input_text, file_path, progress, task_id, text_inference):
    """Generate description, folder name, and filename for a text document."""
    total_steps = 3

    # Step 1: Generate description
    description = summarize_text_content(input_text, text_inference)
    progress.update(task_id, advance=1 / total_steps)

    # Step 2: Generate filename
    filename_prompt = f"""Based on the summary below, generate a specific and descriptive filename that captures the essence of the document.
Limit the filename to a maximum of 3 words. Use nouns and avoid starting with verbs like 'depicts', 'shows', 'presents', etc.
Do not include any data type words like 'text', 'document', 'pdf', etc. Use only letters and connect words with underscores.

Summary: {description}

Examples:
1. Summary: A research paper on the fundamentals of string theory.
   Filename: string_theory_fundamentals

2. Summary: An article discussing the effects of climate change on polar bears.
   Filename: climate_polar_bears

Now generate the filename.

Output only the filename, without any additional text.

Filename:"""
    filename = text_inference(filename_prompt)
    filename = re.sub(r'^Filename:\s*', '', filename, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    # Step 3: Generate folder name from summary
    foldername_prompt = f"""Based on the summary below, generate a general category or theme that best represents the main subject of this document.
This will be used as the folder name. Limit the category to a maximum of 2 words. Use nouns and avoid verbs.
Do not include specific details, words from the filename, or any generic terms like 'untitled' or 'unknown'.

Summary: {description}

Examples:
1. Summary: A research paper on the fundamentals of string theory.
   Category: physics

2. Summary: An article discussing the effects of climate change on polar bears.
   Category: environment

Now generate the category.

Output only the category, without any additional text.

Category:"""
    foldername = text_inference(foldername_prompt)
    foldername = re.sub(r'^Category:\s*', '', foldername, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    sanitized_filename = sanitize_filename(filename, max_words=3)
    sanitized_foldername = sanitize_filename(foldername, max_words=2)

    return sanitized_foldername, sanitized_filename, description