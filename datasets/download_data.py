"""Resumable downloads; a final filename denotes a completed transfer."""
import os
import time
import requests
from tqdm import tqdm


def check_download_size(url):
    try:
        with requests.head(url, allow_redirects=True, timeout=30) as response:
            response.raise_for_status()
            value = response.headers.get('Content-Length')
            return int(value) if value is not None else None
    except requests.RequestException:
        return None


def download_file(url, destination, *, request_input=False, request_headers=None):
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return
    if not url:
        raise ValueError(f'No download URL configured for {destination}; stage this local dataset first.')
    size = check_download_size(url)
    if request_input:
        description = f'{size / 1024**3:.2f} GB' if size is not None else 'size unknown'
        answer = input(f'Download {destination} ({description})? [Y/n] ').strip().lower()
        if answer not in ('', 'y', 'yes'):
            raise RuntimeError('Download cancelled')
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + '.part'
    for attempt in range(5):
        offset = os.path.getsize(partial) if os.path.exists(partial) else 0
        if size is not None and offset == size:
            os.replace(partial, destination)
            return
        if size is not None and offset > size:
            os.unlink(partial)
            offset = 0
        headers = dict(request_headers or {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://deepblue.lib.umich.edu/', 'Accept': '*/*',
        })
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise IOError('Invalid Content-Range in resumed response')
                with open(partial, 'ab' if offset else 'wb') as handle, tqdm(
                    desc=os.path.basename(destination), total=size, initial=offset,
                    unit='B', unit_scale=True) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
            if size is not None and os.path.getsize(partial) != size:
                raise IOError(f'Incomplete download: {destination}')
            os.replace(partial, destination)
            return
        except (requests.RequestException, IOError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download_sequence_data(config, data_config, dataset_name, sequence_name):
    destination = os.path.join(config['data_path'], dataset_name, sequence_name,
                               f"{sequence_name}.{data_config['format']['data']['format']}")
    download_file(data_config['sequences'][sequence_name]['data']['url'], destination,
                  request_input=config.get('request_input', False))
