# Deployment Instructions for Day 5B: ADK Agent to Vertex AI Agent Engine

## ⚠️ Cost and Safety Information

- **Free Tier Eligible**: This deployment uses Vertex AI Agent Engine which offers a monthly free tier
- **Automatic Cleanup**: The script automatically deletes the deployed agent when finished to prevent ongoing charges
- **Billing Required**: You need to enable billing on your Google Cloud account (required even for free tier)
- **Monitor Usage**: Keep an eye on your usage to stay within free tier limits
- **Interrupt Risk**: If the script is interrupted before cleanup, you may incur charges - check the console to manually delete if needed
- **Original Warning**: The original notebook warns to "Always delete resources when done testing!" since leaving agents running can incur costs

This guide provides step-by-step instructions to deploy your ADK agent to Vertex AI Agent Engine using the Day5b.py script.

## Prerequisites

Before you can deploy your agent, you need to set up your Google Cloud environment:

### 1. Google Cloud Account
- Create a free Google Cloud account at [https://cloud.google.com/free](https://cloud.google.com/free)
- New users get $300 in free credits valid for 90 days
- Enable billing on your account (a credit card is needed for verification)

### 2. Google Cloud Project
- Go to the [Google Cloud Console](https://console.cloud.google.com/)
- Create a new project or select an existing one
- Note down your Project ID (it usually looks like `my-project-12345`)

### 3. Enable Required APIs
Enable the following APIs in your Google Cloud Console:
- Vertex AI API
- Cloud Storage API
- Cloud Logging API
- Cloud Monitoring API
- Cloud Trace API
- Telemetry API

You can [use this link](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,storage.googleapis.com,logging.googleapis.com,monitoring.googleapis.com,cloudtrace.googleapis.com,telemetry.googleapis.com) to open the Google Cloud Console and enable these APIs.

### 4. Install Google Cloud SDK
- Install the Google Cloud SDK by following instructions at [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
- Authenticate using: `gcloud auth login`
- Set your project: `gcloud config set project YOUR_PROJECT_ID`

### 5. Set Up API Credentials
Add the following to your `.env` file in the root directory of this project:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=global
```

Replace `your_api_key_here` with your actual Google API key and `your_project_id` with your actual Google Cloud Project ID.

## Environment Configuration

### 1. Set Up Environment Variables
Add the following to your `.env` file in the root directory of this project:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=global
```

Replace `your_api_key_here` with your actual Google API key and `your_project_id` with your actual Google Cloud Project ID.

### 2. Verify Setup
Before deploying, verify your setup:

```bash
# Check that gcloud is authenticated
gcloud auth list

# Check that your project is set
gcloud config list project

# Test API access
gcloud projects list
```

## Running the Deployment Script

### 1. Install Dependencies
Make sure you have all dependencies installed:

```bash
python3 -m pip install -r prerequisites.py
# Or run the prerequisites script:
python3 prerequisites.py
```

### 2. Run the Deployment Script
```bash
python3 Day5b.py
```

## Expected Deployment Process

The script will perform the following steps:
1. Create the necessary agent files in the `sample_agent/` directory
2. Deploy the agent to Vertex AI Agent Engine (this takes 2-5 minutes)
3. Wait for deployment to complete
4. Retrieve the deployed agent and test it with a sample query
5. Automatically clean up the deployed agent when finished

## Monitoring Your Deployment

### Check Deployment Status in Console
- Visit the [Vertex AI Agent Engine Console](https://console.cloud.google.com/vertex-ai/agents/agent-engines)
- You can see your deployed agents and their status

### Check Costs
- Visit the [Google Cloud Billing Console](https://console.cloud.google.com/billing)
- Monitor usage related to Vertex AI Agent Engine

## Troubleshooting

### Common Issues

1. **Authentication Error**: Make sure you've authenticated with `gcloud auth login`
2. **Permission Error**: Ensure your service account has the necessary permissions (Vertex AI User role)
3. **API Not Enabled**: Verify that all required APIs are enabled in your project
4. **Project ID Not Found**: Double-check that your project ID is correct

### Required IAM Permissions
Your service account needs these roles:
- Vertex AI User
- Service Account Token Creator (if using service account authentication)

## Important: Cleanup and Cost Management

⚠️ **IMPORTANT: Always clean up resources when done testing to avoid unexpected charges!**

The Day5b.py script is designed to automatically delete the deployed agent when it completes, but if the script fails or is interrupted, you should manually delete resources:

1. Go to [Vertex AI Agent Engine Console](https://console.cloud.google.com/vertex-ai/agents/agent-engines)
2. Find and delete your deployed agent

## Testing Your Deployment (Alternative Method)

If you want to test without running the full script, you can manually verify your setup:

```bash
# Test gcloud authentication
gcloud auth print-access-token

# Test ADK CLI availability
adk --help

# Verify project access
gcloud projects describe YOUR_PROJECT_ID
```

## Region Selection

The script will randomly select from these regions:
- europe-west1
- europe-west4
- us-east4
- us-west1

For production, choose a region close to your users for lower latency.

## Memory Bank Configuration (Optional)

To add long-term memory capabilities:
1. Add memory tools to your agent code (PreloadMemoryTool, LoadMemoryTool)
2. Add a callback to save conversations to Memory Bank
3. Redeploy your agent

For more details, refer to the "Long-Term Memory with Vertex AI Memory Bank" section in the original notebook.