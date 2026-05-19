# Downloading Waymo Open Motion Dataset to SJSU HPC Scratch

This guide provides steps to download large datasets (2.5TB+) from Google Cloud Platform (GCP) buckets directly to the `/scratch/cmpe257-fa26` directory without using space on the login node's home directory.

## 1. Set Up Workspace on Scratch
Since the home directory has strict storage limits, all tools, configurations, and data must live on the scratch partition.

```bash
# Navigate to the target scratch directory
cd /scratch/cmpe257-fa26

# Create a directory for tools and configuration
mkdir -p my_gcloud/config

# Redirect the Google Cloud SDK config path to scratch
export CLOUDSDK_CONFIG=/scratch/cmpe257-fa26/my_gcloud/config
```

## 2. Install Google Cloud SDK on Scratch
Download and install the `gsutil` tool directly into the scratch folder.

```bash
# Download the SDK
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz

# Extract and install
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --path-update=false --usage-reporting=false --command-completion=false
```

## 3. Authenticate with Waymo Access
Link the SDK to your Google account that has permissions for the Waymo dataset.

```bash
# Initialize login
./google-cloud-sdk/bin/gcloud auth login --no-launch-browser
```
1. Copy the generated URL into your local computer's browser.
2. Log in and authorize.
3. Paste the verification code back into the terminal.

## 4. Run the Download (Background Session)
To prevent the download from failing if your SSH connection drops, use `screen`.

```bash
# Start a named background session
screen -S waymo_download

# (Inside screen) Ensure config path is still set
export CLOUDSDK_CONFIG=/scratch/cmpe257-fa26/my_gcloud/config

# Start the multi-threaded download (modify the GS path for your specific version)
./google-cloud-sdk/bin/gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/" .
```

*   **Detach from session:** Press `Ctrl + A`, then `D`.
*   **Reconnect later:** Run `screen -r waymo_download`.

## 5. Important Notes
- **Network:** Ensure you are on the [SJSU VPN](https://sjsu.edu) or **SJSU_premier WiFi**.
- **Quota:** If you receive a "Permission Denied" or "Disk Quota Exceeded" error, verify that `CLOUDSDK_CONFIG` is pointing to `/scratch`.
- **Cleanup:** Once the download is complete, you can remove the `.tar.gz` file to save space.
