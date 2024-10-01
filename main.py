import os
import time
from dotenv import load_dotenv
from llm_utils import get_llm_response, get_text_llm, get_vision_llm
import groq
import requests

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_file_data
)

from data_processing_common import (
    compute_operations,
    execute_operations,
    process_files_by_date,
    process_files_by_type,
)

from text_data_processing import (
    process_text_files
)

from image_data_processing import (
    process_single_image,
    process_image_files
)

from output_filter import filter_specific_output  # Import the context manager

def ensure_nltk_data():
    """Ensure that NLTK data is downloaded efficiently and quietly."""
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Load environment variables
load_dotenv()

# Initialize GROQ client for vision tasks
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize DeepInfra client for text tasks
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = get_text_llm("deepinfra")

# Initialize DeepSeek client for text tasks
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = get_text_llm("deepseek")

def deepinfra_chat_completion(prompt):
    return get_llm_response(DEEPINFRA_MODEL, prompt)

def deepseek_chat_completion(prompt):
    return get_llm_response(DEEPSEEK_MODEL, prompt)

def initialize_models(text_llm_provider, vision_llm_provider):
    """Initialize the models if they haven't been initialized yet."""
    print("**----------------------------------------------**")
    print(f"**       Text LLM (DeepInfra): {DEEPINFRA_MODEL}**")
    print(f"**       Text LLM (DeepSeek): {DEEPSEEK_MODEL}  **")
    print(f"**       Vision LLM: {get_vision_llm(vision_llm_provider)}         **")
    print("**----------------------------------------------**")

# Load environment variables
load_dotenv()

# Initialize GROQ client
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

def simulate_directory_tree(operations, base_path):
    """Simulate the directory tree based on the proposed operations."""
    tree = {}
    for op in operations:
        rel_path = os.path.relpath(op['destination'], base_path)
        parts = rel_path.split(os.sep)
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    return tree

def print_simulated_tree(tree, prefix=''):
    """Print the simulated directory tree."""
    pointers = ['├── '] * (len(tree) - 1) + ['└── '] if tree else []
    for pointer, key in zip(pointers, tree):
        print(prefix + pointer + key)
        if tree[key]:  # If there are subdirectories or files
            extension = '│   ' if pointer == '├── ' else '    '
            print_simulated_tree(tree[key], prefix + extension)

def get_yes_no(prompt):
    """Prompt the user for a yes/no response."""
    while True:
        response = input(prompt).strip().lower()
        if response in ('yes', 'y'):
            return True
        elif response in ('no', 'n'):
            return False
        elif response == '/exit':
            print("Exiting program.")
            exit()
        else:
            print("Please enter 'yes' or 'no'. To exit, type '/exit'.")

def get_mode_selection():
    """Prompt the user to select a mode."""
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content")
        print("2. By Date")
        print("3. By Type")
        response = input("Enter 1, 2, or 3 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return 'content'
        elif response == '2':
            return 'date'
        elif response == '3':
            return 'type'
        else:
            print("Invalid selection. Please enter 1, 2, or 3. To exit, type '/exit'.")

def get_text_llm_selection():
    """Prompt the user to select an LLM for text processing."""
    while True:
        print("Please choose the LLM for text processing:")
        print("1. DeepInfra (Qwen/Qwen2.5-72B-Instruct)")
        print("2. DeepSeek")
        response = input("Enter 1 or 2 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return "deepinfra"
        elif response == '2':
            return "deepseek"
        else:
            print("Invalid selection. Please enter 1 or 2. To exit, type '/exit'.")

def get_vision_llm_selection():
    """Prompt the user to select an LLM for vision processing."""
    while True:
        print("Please choose the LLM for vision processing:")
        print("1. GROQ (llama-3.2-11b-vision-preview)")
        print("2. OpenAI (GPT-4 Vision)")
        response = input("Enter 1 or 2 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return "groq"
        elif response == '2':
            return "openai"
        else:
            print("Invalid selection. Please enter 1 or 2. To exit, type '/exit'.")

def get_text_llm_wrapper(text_llm_provider):
    """Wrapper function to call get_llm_response with the correct arguments for text processing."""
    def wrapper(prompt):
        text_model = get_text_llm(text_llm_provider)
        return get_llm_response(text_model, prompt, provider=text_llm_provider)
    return wrapper

def main():
    # Ensure NLTK data is downloaded efficiently and quietly
    ensure_nltk_data()

    # Start with dry run set to True
    dry_run = True

    # Display silent mode explanation before asking
    print("-" * 50)
    print("**NOTE: Silent mode logs all outputs to a text file instead of displaying them in the terminal.")
    silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
    if silent_mode:
        log_file = 'operation_log.txt'
    else:
        log_file = None

    while True:
        # Paths configuration
        if not silent_mode:
            print("-" * 50)

        # Get input and output paths once per directory
        input_path = input("Enter the path of the directory you want to organize: ").strip()
        while not os.path.exists(input_path):
            message = f"Input path {input_path} does not exist. Please enter a valid path."
            if silent_mode:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
            input_path = input("Enter the path of the directory you want to organize: ").strip()

        # Confirm successful input path
        message = f"Input path successfully uploaded: {input_path}"
        if silent_mode:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)

        # Default output path is a folder named "organized_folder" in the same directory as the input path
        output_path = input("Enter the path to store organized files and folders (press Enter to use 'organized_folder' in the input directory): ").strip()
        if not output_path:
            # Get the parent directory of the input path and append 'organized_folder'
            output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

        # Confirm successful output path
        message = f"Output path successfully set to: {output_path}"
        if silent_mode:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)

        # Start processing files
        start_time = time.time()
        file_paths = collect_file_paths(input_path)
        end_time = time.time()

        message = f"Time taken to collect file paths: {end_time - start_time:.2f} seconds"
        if silent_mode:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)
            print("Directory tree before organizing:")
            display_directory_tree(input_path)

            print("*" * 50)

        # Loop for selecting sorting methods
        while True:
            mode = get_mode_selection()

            if mode == 'content':
                # Proceed with content mode
                # Initialize models once
                if not silent_mode:
                    print("Initializing AI models for processing...")
                
                text_llm_provider = get_text_llm_selection()
                vision_llm_provider = get_vision_llm_selection()
                
                initialize_models(text_llm_provider, vision_llm_provider)

                if not silent_mode:
                    print("*" * 50)
                    print("File paths collected successfully. Processing may take a few minutes.")
                    print("*" * 50)

                # Prepare to collect link type statistics
                link_type_counts = {'hardlink': 0, 'symlink': 0}

                # Separate files by type
                image_files, text_files = separate_files_by_type(file_paths)

                # Create the text_llm_wrapper with the selected provider
                text_llm_wrapper = get_text_llm_wrapper(text_llm_provider)

                # Prepare text tuples for processing
                text_tuples = []
                for fp in text_files:
                    # Use read_file_data to read the file content
                    text_content = read_file_data(fp, text_llm_wrapper)
                    if text_content is None:
                        message = f"Unsupported or unreadable text file format: {fp}"
                        if silent_mode:
                            with open(log_file, 'a') as f:
                                f.write(message + '\n')
                        else:
                            print(message)
                        continue  # Skip unsupported or unreadable files
                    text_tuples.append((fp, text_content))

                # Process files sequentially
                data_images = process_image_files(image_files, groq_client, vision_llm_provider, silent=silent_mode, log_file=log_file)

                data_texts = process_text_files(text_tuples, text_llm_wrapper, silent=silent_mode, log_file=log_file)

                # Prepare for copying and renaming
                renamed_files = set()
                processed_files = set()

                # Combine all data
                all_data = data_images + data_texts

                # Compute the operations
                operations = compute_operations(
                    all_data,
                    output_path,
                    renamed_files,
                    processed_files,
                    client  # Pass the client to compute_operations
                )

            elif mode == 'date':
                # Process files by date
                operations = process_files_by_date(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
            elif mode == 'type':
                # Process files by type
                operations = process_files_by_type(file_paths, output_path, dry_run=False, silent=silent_mode, log_file=log_file)
            else:
                print("Invalid mode selected.")
                return

            # Simulate and display the proposed directory tree
            print("-" * 50)
            message = "Proposed directory structure:"
            if silent_mode:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
                print(os.path.abspath(output_path))
                simulated_tree = simulate_directory_tree(operations, output_path)
                print_simulated_tree(simulated_tree)
                print("-" * 50)

            # Ask user if they want to proceed
            proceed = get_yes_no("Would you like to proceed with these changes? (yes/no): ")
            if proceed:
                # Create the output directory now
                os.makedirs(output_path, exist_ok=True)

                # Perform the actual file operations
                message = "Performing file operations..."
                if silent_mode:
                    with open(log_file, 'a') as f:
                        f.write(message + '\n')
                else:
                    print(message)
                execute_operations(
                    operations,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )

                message = "The files have been organized successfully."
                if silent_mode:
                    with open(log_file, 'a') as f:
                        f.write("-" * 50 + '\n' + message + '\n' + "-" * 50 + '\n')
                else:
                    print("-" * 50)
                    print(message)
                    print("-" * 50)
                break  # Exit the sorting method loop after successful operation
            else:
                # Ask if the user wants to try another sorting method
                another_sort = get_yes_no("Would you like to choose another sorting method? (yes/no): ")
                if another_sort:
                    continue  # Loop back to mode selection
                else:
                    print("Operation canceled by the user.")
                    break  # Exit the sorting method loop

        # Ask if the user wants to organize another directory
        another_directory = get_yes_no("Would you like to organize another directory? (yes/no): ")
        if not another_directory:
            break  # Exit the main loop


if __name__ == '__main__':
    main()