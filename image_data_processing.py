import re
import os
import time
import base64
from PIL import Image
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from data_processing_common import sanitize_filename
from llm_utils import get_vision_llm
import groq

def is_animated_gif(image_path):
    try:
        with Image.open(image_path) as img:
            return getattr(img, "is_animated", False)
    except:
        return False

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_single_image(image_path, groq_client, vision_llm_provider, silent=False, log_file=None):
    """Process a single image file to generate metadata."""
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(image_path)}", total=1.0)
        
        if is_animated_gif(image_path):
            foldername = "animated_gifs"
            filename = os.path.basename(image_path)
            description = "Animated GIF (not processed by AI)"
        else:
            foldername, filename, description = generate_image_metadata(image_path, progress, task_id, groq_client, vision_llm_provider)
    
    end_time = time.time()
    time_taken = end_time - start_time

    message = f"File: {image_path}\nTime taken: {time_taken:.2f} seconds\nDescription: {description}\nFolder name: {foldername}\nGenerated filename: {filename}\n"
    if silent:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
    else:
        print(message)
    return {
        'file_path': image_path,
        'foldername': foldername,
        'filename': filename,
        'description': description
    }

def process_image_files(image_files, groq_client, vision_llm_provider, silent=False, log_file=None):
    """Process image files sequentially."""
    results = []
    for image_file in image_files:
        try:
            data = process_single_image(image_file, groq_client, vision_llm_provider, silent=silent, log_file=log_file)
            results.append(data)
        except Exception as e:
            message = f"Error processing image file {image_file}: {str(e)}"
            if silent:
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(message + '\n')
            else:
                print(message)
    return results

def generate_image_metadata(image_path, progress, task_id, groq_client, vision_llm_provider):
    """Generate description, folder name, and filename for an image file."""
    total_steps = 3
    
    # Encode the image
    base64_image = encode_image(image_path)

    vision_model = get_vision_llm(vision_llm_provider)

    # Step 1: Generate description using Vision LLM
    description_prompt = "Please provide a detailed description of this image, focusing on the main subject and any important details."
    description_response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=vision_model,
    )
    description = description_response.choices[0].message.content.strip()
    progress.update(task_id, advance=1 / total_steps)

    # Step 2: Generate filename using Vision LLM
    filename_prompt = f"""Based on the description below, generate a specific and descriptive filename for the image.
    Limit the filename to a maximum of 3 words. Use nouns and avoid starting with verbs like 'depicts', 'shows', 'presents', etc.
    Do not include any data type words like 'image', 'jpg', 'png', etc. Use only letters and connect words with underscores.

    Description: {description}

    Example:
    Description: A photo of a sunset over the mountains.
    Filename: sunset_over_mountains

    Now generate the filename.

    Output only the filename, without any additional text.

    Filename:"""
    filename_response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": filename_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=vision_model,
    )
    filename = filename_response.choices[0].message.content.strip()
    filename = re.sub(r'^Filename:\s*', '', filename, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    # Step 3: Generate folder name from description using Vision LLM
    foldername_prompt = f"""Based on the description below, generate a general category or theme that best represents the main subject of this image.
    This will be used as the folder name. Limit the category to a maximum of 2 words. Use nouns and avoid verbs.
    Do not include specific details, words from the filename, or any generic terms like 'untitled' or 'unknown'.

    Description: {description}

    Examples:
    1. Description: A photo of a sunset over the mountains.
       Category: landscapes

    2. Description: An image of a smartphone displaying a storage app with various icons and information.
       Category: technology

    3. Description: A close-up of a blooming red rose with dew drops.
       Category: nature

    Now generate the category.

    Output only the category, without any additional text.

    Category:"""
    foldername_response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": foldername_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=vision_model,
    )
    foldername = foldername_response.choices[0].message.content.strip()
    foldername = re.sub(r'^Category:\s*', '', foldername, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    sanitized_filename = sanitize_filename(filename, max_words=3)
    sanitized_foldername = sanitize_filename(foldername, max_words=2)

    return sanitized_foldername, sanitized_filename, description