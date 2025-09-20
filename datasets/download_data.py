# datasets/data_utils.py
import os, requests, time
from tqdm import tqdm

def download_sequence_data(config, data_config, dataset_name, sequence_name):
    # Get the URL for the sequence data
    sequence_url = data_config['sequences'][sequence_name]['data']['url']
    # Validate the URL
    size = check_download_size(sequence_url)
    # Check with the user if they want to download the dataset
    if config['request_input']:
        user_response = input(f"Download the sequence data ({int(size/(1024*1024*1024))}GB)? (yes / press Enter to confirm): ").strip().lower()
        if user_response not in ('', 'yes'):
            raise Exception("Download cancelled by user.")
    
    # Create directory if they do not exist
    os.makedirs(os.path.join(config['data_path'], dataset_name, sequence_name), exist_ok=True)
    
    # Use sequence name with correct format from config
    file_format = data_config['format']['data']['format']
    filename = f"{sequence_name}.{file_format}"
    
    # Full file path
    file_path = os.path.join(config['data_path'], dataset_name, sequence_name, filename)
    
    # Download with resume capability and retry logic
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Check if file already exists (partial download)
            resume_pos = 0
            if os.path.exists(file_path):
                resume_pos = os.path.getsize(file_path)
                if resume_pos == size:
                    print(f"File already completely downloaded: {file_path}")
                    return
                elif resume_pos > 0:
                    print(f"Resuming download from {resume_pos / (1024*1024):.1f}MB")
            
            # Set up headers for resume
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            # Make the actual HTTP GET request
            http_response = requests.get(sequence_url, headers=headers, stream=True, timeout=30)
            http_response.raise_for_status()
            
            # Open file in appropriate mode
            mode = 'ab' if resume_pos > 0 else 'wb'
            
            with open(file_path, mode) as file, tqdm(
                desc=f"Download sequence data = {sequence_name}",
                total=size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in http_response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Download completed: {file_path}")
            return  # Success, exit retry loop
            
        except (requests.exceptions.RequestException, 
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            
            print(f"Download failed on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Download failed after {max_retries} attempts")
                # Clean up partial download if it's corrupted
                if os.path.exists(file_path):
                    current_size = os.path.getsize(file_path)
                    print(f"Partial file size: {current_size / (1024*1024):.1f}MB")
                    print("You can retry the download to resume from this point")
                raise
        
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
            current_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"Partial file saved: {current_size / (1024*1024):.1f}MB")
            print("You can restart the download to resume from this point")
            raise

def check_download_size(sequence_url):
    try:
        response = requests.head(sequence_url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        # Get the Content-Length header
        size_header = response.headers.get('Content-Length')
        if size_header:
            return int(size_header)
        else:
            print("Warning: Server did not provide file size")
            return None
        
    except requests.RequestException as e:
        print(f"Error checking file size: {e}")
        return None