import os
import json
import shutil
import hashlib
import csv
from urllib.parse import urlparse
from cve_bin_tool.cli import main as cve_main
import xml.etree.ElementTree as ET

CSV_FILE_PATH = '/var/task/repo_archive_list.csv'

def find_bom_files(directory):
    bom_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_element = tree.getroot()
                    # Get the tag without the namespace
                    tag = root_element.tag.split('}', 1)[-1]
                    # Check for BOM files
                    if tag == 'bom' or tag == 'BOM':
                        bom_files.append(file_path)
                    # Check for SoftwareIdentity files
                    elif tag == 'SoftwareIdentity':
                        bom_files.append(file_path)
                except ET.ParseError:
                    continue
    return bom_files

def read_urls_from_csv(file_path):
    url_list = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                url_list.append(row[0])
    return url_list

def get_unique_url(request_id, url_list):
    # Hash the request ID
    hash_object = hashlib.md5(request_id.encode())
    hash_digest = hash_object.hexdigest()

    # Convert the hash to an integer
    hash_int = int(hash_digest, 16)

    # Compute the index using modulo operation
    index = hash_int % len(url_list)

    return url_list[index]

# Function to get repository name from zip URL
def get_repo_dir_from_zip_url(zip_url):
    parsed_url = urlparse(zip_url)
    path_parts = parsed_url.path.split('/')
    repo_name = path_parts[2]  # The repository name is the third part of the path
    branch_name = path_parts[6].split('.')[0]
    return repo_name+"-"+branch_name

# Function to run cve-bin-tool using its Python API
def run_cve_bin_tool(repo_path):
    results = {}
    if os.path.exists(repo_path):
        print(f"Repository path exists, now running cve-bin-tool on {repo_path}")

    bom_files = find_bom_files(repo_path)
    if bom_files:
        print("BOM files found:")
        for bom_file in bom_files:
            print(bom_file)
            # vulnerabilities = cve_main(['cve-bin-tool --offline --disable-version-check --disable-validation-check -r openssl -s all --exclude *.jpg,*.jpeg,*.png,*.gif,*.svg,*.md,*.txt,*.rst,tests,__pycache__,.pytest_cache,*.yml,*.yaml,*.json,*.toml,build,dist,*.egg-info,.eggs,.tox,.venv,venv', bom_file])
            vulnerabilities = cve_main([
                'cve-bin-tool', 
                '--disable-version-check', 
                '--disable-validation-check', 
                '--offline', 
                '-r', 'openssl',  # Only run the openssl checker
                bom_file
            ])
            results[os.path.basename(bom_file)] = vulnerabilities
    else:
        print("No BOM files found.")

    # vulnerabilities = cve_main(['cve-bin-tool --offline --disable-version-check --disable-validation-check -r openssl -s all --exclude *.jpg,*.jpeg,*.png,*.gif,*.svg,*.md,*.txt,*.rst,tests,__pycache__,.pytest_cache,*.yml,*.yaml,*.json,*.toml,build,dist,*.egg-info,.eggs,.tox,.venv,venv', repo_path])
    vulnerabilities = cve_main([
        'cve-bin-tool', 
        '--disable-version-check', 
        '--disable-validation-check', 
        '--offline', 
        '-r', 'openssl',  # Only run the openssl checker
        repo_path
    ])
    results[os.path.basename(repo_path)] = vulnerabilities
    return results

def copy_dependencies(src, dest, name):
    if os.path.exists(src) and not os.path.exists(dest):
        print(f"The directory {dest} doesn't exist, copying {name} from {src} to {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)

def handler(event, context=None):
    # Paths
    task_root = os.getenv('LAMBDA_TASK_ROOT', '/var/task')
    cvedb_path_root = os.path.join(task_root, '.cache')
    repos_path_root = os.path.join(task_root, 'cloned_repositories')
    repos_path_tmp = '/tmp/cloned_repositories'
    cvedb_path_tmp = '/tmp/.cache'

    # Ensure the repo ephemeral destination directory exists
    os.makedirs(repos_path_tmp, exist_ok=True)

    # Copy CVE Database
    copy_dependencies(cvedb_path_root, cvedb_path_tmp, '.cache')

    # Read the URLs from the CSV file
    url_list = read_urls_from_csv(CSV_FILE_PATH)
    request_id = context.aws_request_id if context else "default_request_id"
    repo_url = get_unique_url(request_id, url_list)

    # Configure repository paths
    repo_dir = get_repo_dir_from_zip_url(repo_url)
    repo_path_root = os.path.join(repos_path_root, repo_dir)
    repo_path_tmp = os.path.join(repos_path_tmp, repo_dir)

    # Copy repository if needed
    copy_dependencies(repo_path_root, repo_path_tmp, repo_dir)

    # print(f"Going to run CVE Tool on the repo {repo_path}")
    scan_results = run_cve_bin_tool(repo_path_tmp)

    return {
        'statusCode': 200,
        'body': json.dumps(scan_results)
    }

if __name__ == "__main__":
    event = {
            'repo_url': 'https://github.com/bitcoin-kyrgyzstan/btc-kgs-pos/archive/refs/heads/main.zip'
    }
    print(handler({'body': json.dumps(event)}))
