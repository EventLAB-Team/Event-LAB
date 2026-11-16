"""Download dataset assets without constructing a runtime dataset."""
import os, requests, re, hashlib
import yaml
from pathlib import Path
from urllib.parse import unquote
from datasets.download_data import download_sequence_data, download_file


def sequence_download_status(config, data_config, dataset_name, sequence_name):
    """Inspect final files only; no network requests, parsing, or dataset construction."""
    folder = Path(config['data_path']) / dataset_name / sequence_name
    sequence = data_config['sequences'][sequence_name]
    assets = [(folder / f"{sequence_name}.{data_config['format']['data']['format']}", sequence['data']['url'])]
    ground_truth = sequence['ground_truth']
    if ground_truth['available']:
        assets.append((folder / f"{sequence_name}_ground_truth.{data_config['format']['ground_truth']}", ground_truth['url']))
    if ground_truth.get('sync_url'):
        assets.append((folder / 'sync.json', ground_truth['sync_url']))
    missing = [(path, url) for path, url in assets if not path.is_file() or path.stat().st_size == 0]
    if not missing:
        return 'Downloaded'
    if any(not url for _, url in missing):
        return 'Local only'
    if len(missing) < len(assets) or any(path.exists() or Path(str(path) + '.part').exists() for path, _ in assets):
        return 'Partial'
    return 'Missing'


class DatasetFiles:
    def __init__(self, config, dataset_name, sequence_name):
        self.config = config # General configuration
        self.dataset_name = dataset_name # Name of the dataset
        self.sequence_name = sequence_name # Name of the sequence

        # Load dataset configuration
        with open(f"./datasets/{self.dataset_name}.yaml", 'r') as file:
            self.data_config = yaml.safe_load(file)

        # Generate dataset name
        self.dataset_path = os.path.join(self.config['data_path'], # data path to the dataset
                                    self.dataset_name, # name of the dataset
                                    self.sequence_name, # the sequence name
                                    )
        # Create dataset directory
        os.makedirs(self.dataset_path, exist_ok=True)

        self.dataset_raw = os.path.join(self.dataset_path,f"{self.sequence_name}.{self.data_config['format']['data']['format']}")

        # EventCV filters hot pixels; the legacy hot-pixel files are not needed.
        download_sequence_data(config, self.data_config, dataset_name, sequence_name)
        ground_truth = self.data_config['sequences'][sequence_name]['ground_truth']
        if ground_truth['available']:
            extension = self.data_config['format']['ground_truth']
            destination = os.path.join(self.dataset_path, f'{sequence_name}_ground_truth.{extension}')
            if not os.path.isfile(destination) or os.path.getsize(destination) == 0:
                url = ground_truth['url']
                checksum = None
                with requests.Session() as session:
                    session.headers.update(self._download_headers())
                    if 'zenodo' in url:
                        url, _, checksum = self._resolve_zenodo_file(url, session)
                download_file(url, destination, request_input=config.get('request_input', False),
                              request_headers=self._download_headers())
                if checksum and checksum.startswith('md5:'):
                    if self._md5(destination) != checksum.split(':', 1)[1]:
                        os.unlink(destination)
                        raise IOError(f'Ground truth checksum mismatch: {destination}')

        if ground_truth.get('sync_url'):
            download_file(ground_truth['sync_url'], os.path.join(self.dataset_path, 'sync.json'),
                          request_input=config.get('request_input', False))

    def _download_headers(self):
        headers = {
            "User-Agent": "EventLAB/1.0 (+https://github.com/EventLAB-Team/Event-LAB)",
            "Accept": "application/octet-stream,application/json;q=0.9,*/*;q=0.8",
        }
        token = os.getenv("ZENODO_API_TOKEN") or os.getenv("ZENODO_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


    def _resolve_zenodo_file(self, url, session):
        """
        Resolve a Zenodo pretty file URL to the actual downloadable API file URL,
        and return (download_url, size, checksum).
        If it's not a Zenodo records/files URL, return the input URL unchanged.
        """
        ZENODO_FILE_RE = re.compile(r"^https?://(?:www\.)?zenodo\.org/records?/(\d+)/files/([^/?#]+)")
        m = ZENODO_FILE_RE.match(url)
        if not m:
            return url, None, None

        record_id = m.group(1)
        wanted_name = unquote(m.group(2))

        meta_url = f"https://zenodo.org/api/records/{record_id}"
        r = session.get(meta_url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        meta = r.json()

        files = meta.get("files", [])
        file_entry = None
        for f in files:
            key = f.get("key") or f.get("filename")
            if key == wanted_name:
                file_entry = f
                break

        if file_entry is None:
            available = [f.get("key") or f.get("filename") for f in files]
            raise FileNotFoundError(
                f"Zenodo record {record_id} does not contain '{wanted_name}'. "
                f"Available files: {available}"
            )

        links = file_entry.get("links", {})
        download_url = (
            links.get("self")
            or links.get("content")
            or links.get("download")
        )
        if not download_url:
            raise RuntimeError(
                f"Could not resolve a downloadable link for '{wanted_name}' "
                f"from Zenodo record {record_id}"
            )

        size = file_entry.get("size")
        checksum = file_entry.get("checksum")
        return download_url, size, checksum


    def _md5(self, path, chunk_size=1024 * 1024):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    
