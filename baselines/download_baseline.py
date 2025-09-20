import subprocess, requests

def clone_repo(url, destination=None):
    # Clone a Git repository from the given URL
    if not url.startswith("http"):
        print("Invalid URL. Make sure it starts with http or https.")
        return False

    # Check if URL is reachable
    try:
        response = requests.head(url, timeout=5)
        if response.status_code >= 400:
            print(f"Repository URL returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Failed to connect to the repository URL: {e}")
        return False

    # Prepare the git clone command, run recursive is url has "VPR-methods-evaluation" in it
    if "VPR-methods-evaluation" in url:
        cmd = ["git", "clone", "--recursive", url]
    else:
        cmd = ["git", "clone", url]
    if destination:
        cmd.append(destination)

    try:
        subprocess.run(cmd, check=True)
        print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False