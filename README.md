---
name: Azure Multimodal AI & LLM Processing Accelerator (Python)
description: Build data processing pipelines with Azure AI Services + LLMs
languages:
- python
- bicep
- azdeveloper
products:
- azure-openai
- document-intelligence
- azure-speech
- azure-app-service
- azure-functions
- azure-storage-accounts
- azure-key-vault
- azure
page_type: sample
urlFragment: multimodal-ai-llm-processing-accelerator

---
<!-- YAML front-matter schema: https://review.learn.microsoft.com/en-us/help/contribute/samples/process/onboarding?branch=main#supported-metadata-fields-for-readmemd -->

# Azure Multimodal AI + LLM Processing Accelerator

##### Table of Contents

- [Azure Multimodal AI + LLM Processing Accelerator](#azure-multimodal-ai--llm-processing-accelerator)
  - [Overview](#overview)
    - [Solution Design](#solution-design)
    - [Key features](#key-features)
    - [Why use this accelerator?](#why-use-this-accelerator)
    - [Example pipeline output](#example-pipeline-output)
    - [Prebuilt pipelines](#prebuilt-pipelines)
    - [Common Scenarios and use cases](#common-scenarios--use-cases)
    - [Roadmap & upcoming features](#roadmap--upcoming-features)
    - [FAQ](#faq)
  - [Deployment](#deployment)
    - [Pricing Considerations](#pricing-considerations)
    - [Deploying to Azure with azd](#deploying-to-azure-with-azd)
    - [Running the solution locally](#running-the-solution-locally)
  - [Credits](#credits)
  - [Contributing](#contributing)
  - [Trademarks](#trademarks)

## Overview

This accelerator is as a customizable code template for building and deploying production-grade data processing pipelines that incorporate Azure AI services and Azure OpenAI/AI Studio LLM models. It uses a variety of data pre-processing and enrichment components to make it easy to build complex, reliable and accurate pipelines that solve real-world use cases. If you'd like to use AI to **summarize, classify, extract or enrich** your data with structured and reliable outputs, this is the code repository for you.

#### Important Note: This accelerator is currently under development and may include regular breaking changes

It is recommended to review the main repo before pulling new changes, as work is in progress to replace many of the third-party components (e.g. those imported from Haystack) with more complete, performant & fully-featured components. Once the core application is stable, a standard release pattern with semantic versioning will be used to manage releases.

### Solution Design

![Solution Design](/docs/solution-design.png)

### Key features

- **Backend processing:** A pre-built Azure Function App, along with a number of pre-built processing pipeline blueprints that can be easily modified and deployed.
- **Front-end demo app:** A simple demo web app to enable internal testing of the backend APIs via a UI, and make it easier to collaborate with non-technical users.
- **Data converters and processors:** Many of the core components required for multimodal processing are included, such as [Azure Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence), [Azure AI Speech](https://azure.microsoft.com/en-us/products/ai-services/ai-speech), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) and more. These help you easily convert your data into the format expected by LLMs, and the blueprints are built in an open way so that you can easily incorporate custom-built components.
- **Enriched outputs & confidence scores:** A number of components are included for combining the outputs of pre-processing steps with the LLM outputs. For example, adding confidence scores, bounding boxes, writing styles to the values extracted by the LLM. This allows for reliable automation of tasks instead of having to trust that the LLM is correct (or reviewing every result).
- **Data validation & intermediate outputs:** All pipelines are built to return not just the final result but all intermediate outputs. This lets you reuse the data in other downstream tasks.
- **Powerful and flexible:** The application is built in a way that supports both simple and complicated pipelines. This allows you to build pipelines for <u>all of your use cases</u> without needing to start from scratch with a new code base when you need to something a little more complex. A single deployment can support all of your backend processing pipelines.
- **Infrastructure-as-code:** An Azure Bicep template to deploy the solution and its required components using `azd`.

### Why use this accelerator?

Most organisations have a huge number of simple and tasks and processes that consume large amounts of time and energy. These could be things like classifying and extracting information from documents, summarizing and triaging customer emails, or transcribing and running compliance tasks on contact centre call recordings. While some of these tasks can be automated with existing tools and services, they often require a lot of up-front investment to fully configure and customize in order to have a reliable, working solution. They can also be perform poorly when dealing with input data that is slightly different than expected, and may never be the right fit for scenarios that require the solution to be flexible or adaptable.

On the other hand, Large Language Models have emerged as a powerful and general-purpose approach that is able to handle these complex and varied situations. And more recently, with the move from text-only models to multimodal models that can incorporate text, audio and video, they are a powerful tool that we can use to automate a wide variety of everyday tasks. But while LLMs are powerful and flexible, they have their own shortcomings when it comes to providing precise and reliable outputs, and they too can be sensitive to the quality of raw and unprocessed input data.

| Approach + Examples  | Strengths | Weaknesses |
| ---------|---------|---------|
|**Domain-specific AI models**<br>- OCR<br>- Speech-to-text<br>- Object detection| - Generally better performance on specialized tasks<br>- Consistent performance and output format<br>- Cost-efficient & Scalable | - Outputs may require translation into human-friendly format<br>- Larger up-front development cost<br>- Tuning & customization may be more time-consuming<br>- Customized models may be less reliable/flexible with unseen data|
|**Large Language Models**<br>- Azure OpenAI<br>- Open Source Models | - Define behaviour with natural language<br>- Shorter up-front development time<br>- More flexible with wide-ranging input data<br>- Outputs are in human-friendly format |- Non-deterministic & lower reliability of outputs<br>- Harder to constrain & test (a black box)<br>- No consistent, structured metadata by default<br>- Uncalibrated, no confidence score (when can the outputs be trusted?)<br>- Expensive and slow, especially for transcription/translation tasks |

This accelerator provides the tools and patterns required to combine the best of both worlds in your production workloads, giving you the reasoning power, flexibility and development speed of Large Language Models, while using domain-specific AI during pre and post-processing to increase the consistency, reliability, cost-efficiency of the overall system.

### Example pipeline output

Here is an example of the pre-built [Form Field Extraction pipeline](#prebuilt-pipelines). By combining the structured outputs from Azure Document Intelligence with GPT-4o, we can verify and enrich the values extracted by GPT-4o with confidence scores, bounding boxes, style and more. This allows us to make sure the LLM has not hallucinated, and allows us to automatically flag the document for human review if the confidence scores do not meet our minimum criteria (in this case, all values must have a Document Intelligence confidence score above 80% to avoid human review).

![Form Extraction Example](/docs/form-extraction-example.png)

#### Real-World Case Study

In a recent customer project that involved extracting Order IDs from scanned PDFs and phone images, we used a number of these techniques to increase the performance of GPT-4o-alone from ~60% to near-perfect accuracy:

- Giving GPT-4o the image alone resulted in an overall recall of 60% of the order IDs, and it was impossible to know whether the results for a single PDF could be trusted.
- Adding Document Intelligence text extraction to the pipeline meant the GPT-4o could use analyze both the image and extracted text. This increased the overall recall to over 80%.
- Many images were rotated incorrectly, and GPT-4o performed poorly on these files (around 50% recall). Document Intelligence returns a page rotation value in its response, and using this to correct those images prior to processing by GPT-4o drastically improved performance on those images - from 50% to over 80%.
- Finally, by cross-referencing the order IDs returned by GPT-4o against the Document Intelligence result, we could assign a confidence score to each of the extracted IDs. With these confidence scores, we were able to tune and set confidence thresholds based on the type of image, resulting in 100% recall on 87% of the documents in the dataset, with the rest of the images being automatically sent for human review.

At the conclusion of this project, our customer was able to deploy the solution and automate the majority of their processing workload with confidence, knowing that any cases that were too challenging for the LLM would automatically be escalated for review. Reviews can now be completed in a fraction of the time thanks to the additional metadata returned with each result.

### Prebuilt pipelines

The accelerator comes with these pre-built pipeline examples to help you get started. Each pipeline is built in its own python file as a [function blueprint](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python?tabs=get-started%2Casgi%2Capplication-level&pivots=python-mode-decorators#blueprints), and then imported and added to the main function app within `function_app/function_app.py`.

| Example  | Description & Pipeline Steps|
| ---------|---------|
|**Form Field Extraction with Confidence Scores & bboxes**<br>**(HTTP)**<br>[Code](/function_app/bp_form_extraction_with_confidence.py)| Extracts key information from a PDF form and returns field-level and overall confidence scores and whether human review is required.<br>- PyMuPDF (PDF -> Image)<br>- Document Intelligence (PDF -> text)<br>- GPT-4o (text + image input)<br>- Post-processing:<br><ul>- Match LLM field values with Document Intelligence extracted lines<br>- Merge Confidence scores and bounding boxes<br>- Determine whether to human review is required</ul>- Return structured JSON |
|**Call Center Analysis with Confidence Scores & Timestamps**<br>**(HTTP)**<br>[Code](/function_app/bp_call_center_audio_analysis.py)| Processes a call center recording, classifying customer sentiment & satisfaction, summarizing the call and next best action, and extracting any keywords mentioned. Returns the response with timestamps, confidence scores and the full sentence text for the next best action and each of the keywords mentioned.<br>- Azure AI Speech (Speech -> Text)<br>- GPT-4o (text input)<br>- Post-processing:<br><ul>- Match LLM timestamps to transcribed phrases<br>- Merge sentence info & confidence scores</ul>- Return structured JSON |
|**Form Field Extraction**<br>**(Blob -> CosmosDB)**<br>Code: [Func](/function_app/function_app.py#L19.py), [Pipeline](/function_app/extract_blob_field_info_to_cosmosdb.py)| Summarizes text input into a desired style and number of output sentences.<br>- Pipeline triggered by blob storage event<br>- PyMuPDF (PDF -> Image)<br>- Document Intelligence (PDF -> text)<br>- GPT-4o (text + image input)<br>- Write structured JSON result to CosmosDB container. |
|**Summarize Text**<br>**(HTTP)**<br>[Code](/function_app/bp_summarize_text.py)| Summarizes text input into a desired style and number of output sentences.<br>- GPT-4o (text input + style/length instructions)<br>- Return raw text |
|**City Names Extraction, Doc Intelligence**<br>**(HTTP)**<br>[Code](/function_app/bp_doc_intel_extract_city_names.py)| Uses GPT-4o to extract all city names from a given PDF (using text extracted by Document Intelligence).<br>- Document Intelligence (PDF/image -> text)<br>- GPT-4o (text input)<br>- Return JSON array of city names |
|**City Names Extraction, PyMuPDF**<br>**(HTTP)**<br>[Code](/function_app/bp_pymupdf_extract_city_names.py)| Uses GPT-4o to extract all city names from a given PDF/image + text (extracted locally by PyMuPDF).<br>- PyMuPDF (PDF/image -> text & images)<br>- GPT-4o (text + image input)<br>- Return JSON array of city names |
|**Doc Intelligence Only**<br>**(HTTP)**<br>[Code](/function_app/bp_doc_intel_only.py)| A simple pipeline that calls Document Intelligence and returns the full response and raw text content.<br>- Document Intelligence (PDF/image -> text)<br>- Return structured JSON object |

These pipelines can be duplicated and customized to your specific use case, and should be modified as required. The pipelines all return a large amount of additional information (such as intermediate outputs from each component, time taken for each step, and the raw source code) which will usually not be required in production use cases. Make sure to review the code thoroughly prior to deployment.

### Demo web app

The accelerator comes with an included web app for demo and testing purposes. This webapp is built with [Gradio](https://www.gradio.app/), a lightweight Python UI library, to enable interaction with the backend pipelines from within the browser. The app comes prebuilt with a tab for each of the prebuilt pipelines, along with a few example files for use with each pipeline. The demo app also

### Common scenarios & use cases

- **Call centre analysis:** Transcribe and diarize call centre audio with Azure AI Speech, then use Azure OpenAI to classify the call type, summarize the topics and themes in the call, analyse the sentiment of the customer, and ensure the customer service agent complied with standard procedures (e.g. following the appropriate script, outlining the privacy policy and sending the customer a Product Disclosure Statement).

- **Document processing:** Ingest PDFs, Word documents and scanned images, extract the raw text content with Document Intelligence, then use Azure OpenAI to classify the document by type, extract key fields (e.g. contact information, document ID numbers), classify whether the document was stamped and signed, and return the result in a structured format.
- **Insurance claim processing:** Process all emails and documents in long email chains. Use Azure Document Intelligence to extract information from the attachments, then use Azure OpenAI to generate a timeline of key events in the conversation, determine whether all required documents have been submitted, summarize the current state of the claim, and determine the next-best-action (e.g. auto-respond asking for more information, or escalate to human review for processing).
- **Customer email processing:** Classify incoming emails into categories, summarizing their content, determining the sender's sentiment, and triage into a severity category for human processing.

### Roadmap & upcoming features

This accelerator is in active development, with a list of upcoming features including:

- **Additional Azure AI Services components:** Expand the number of pre-built Azure AI Services components (e.g. Language, Translation and more), while removing dependencies on external libraries where possible.
- **Additional pipeline examples:** A number of additional pipeline examples showcasing other data types/use cases and more advanced pipeline approaches.
- **Evaluation pipelines:** Example evaluation pipelines to evaluate overall performance, making it easy to iterate and improve your processing pipelines and help you select the right production thresholds for accepting a result or escalating to human review.
- **Async processing:** Ensure all included pipeline components have async versions for maximum performance & concurrency.
- **Other input/output options:** Additional pipelines using streaming and websockets.
- **Infrastructure options:** Include options for identity-based authentication between services, private VNets & endpoints etc.
- **Deployment pipeline:** An example deployment pipeline for automated deployments.

To help prioritise these features or request new ones, please head to the Issues section of this repository.

### FAQ

**How can I get started with a solution for my own use case?**

The demo pipelines are examples and require customization in order to have them work accurately in production. The best strategy to get started is to clone one of the existing demo pipelines and modify them for your own purpose. The following steps are recommended:

1. Fork this repository into your own Github account/organization, then clone the repository to your local machine.
1. Follow the instructions in the [deployment section](#deployment) to setup and deploy the code, then test out some of the demo pipelines to understand how they work.
1. Walk through the code for the pipelines that are the most similar to what you would like to build, or which have the different components that you want to use.
    - For example, if you want to build a document extraction pipeline, start with the pipelines that use Azure Document Intelligence.
    - If you want to then combine this with AI Speech or with a different kind of trigger, look through the other pipelines for examples of those.
    - Once familiar with the example pipelines, you should be able to see how you can plug different pipeline components together by into an end-to-end solution.
1. Clone the python blueprint file (e.g. `function_app/bp_<pipeline_name>.py`) that is most similar to your ideal use case, renaming it and using it as a base to start with.
1. Review and modify the different parts of the pipeline. The common things are:
    1. The AI/LLM components that are used and their configurations.
    1. The Azure Function route and required input/output schemas and validation logic.
    1. The Pydantic classes and definitions that define the schema of the LLM's response and the response to be returned from the API.
        - The repo includes a useful Pydantic base model (LLMRawResponseModel) that makes it easy to print out the JSON schema in a prompt-friendly way, and it is suggested to use this model to define your schema so that you can easily provide it to your model and then validate the LLM's responses.
        - By default, these include a lot of additional information from each step of the pipeline, but you may want to remove, modify or add new fields.
    1. The LLM system prompt(s), which contain instructions on how the LLM should complete the task.
        - All prompt examples in this repo are very basic and it is recommended to spend time crafting detailed instructions for the LLM and including some few-shot examples.
        - These should be in addition to the JSON schema definition - if you use the JSON schema alone, expect that the model will make a number of mistakes (you can see this occur in some of the example pipelines).
    1. The post-processing validation logic. This is how you automatically determine when to trust the outputs and when to escalate to human review.
1. Once you have started making progress on the core processing pipeline, you may want to modify the demo web app (`demo_app/`) so that you can easily test the endpoint end-to-end.
    - The Gradio app has a tab built for each of the Function app pipelines, and you should start with the code built for the base of your new function app pipeline.
    - If you need different data inputs or a different request schema (e.g. switching from sending a single file to a file with other JSON parameters), check out each of the other pipelines. These will help you determine how to build the front-end and API request logic so that things work end-to-end.
    - Once you have these working together, you can easily iterate and test your pipelines quickly with the demo web app via the UI.
1. When your pipeline is working end-to-end, it's time to think about testing & evaluating the accuracy and reliability of your solution.
    - It is critical with any AI system to ensure that the pipeline is evaluated on a representative sample of validation data.
    - Without this, it is impossible to know how accurate the solution is, or whether the solution fails under specific circumstances. This is often the time-consuming step of building and deployment an AI solution but is also the most important.
    - While more tools to help simplify this process are coming soon, you should take a look at the [evaluation tools within Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-generative-ai-app).
1. Finally, it's time to deploy your custom application to Azure.
    - Review and modify the infrastructure templates and parameters to ensure the solution is deployed to your requirements
    - Setup automated CI/CD deployment pipelines using Github Actions or Azure DevOps (base templates are coming to the repo soon).

**Does this repo use or support Langchain/Llamaindex/Framework X?**

There are many different frameworks available for LLM/Generative AI applications, each offering different features, integrations, and production suitability. This accelerator uses some existing components from [Haystack](https://haystack.deepset.ai/overview/intro), but it is framework agnostic and you can use any or all frameworks for your pipelines. This allows you to take advantage of the solution architecture and many of the helper functions while still having full control over how you build your pipeline logic.

**What about a custom UI?**

The majority of applications built using this accelerator will be integrated into existing software platforms such as those use in call centres, customer support, case management, ERP platforms and more. Integrating with these platforms typically requires an API call or an event-driven database/blob trigger so that any processing done by this accelerator can seamlessly integrate with any existing workflows and processes (e.g. to trigger escalations, human reviews, automated emails and more).

While a demo application is included in this repository for testing your pipelines, the accelerator is built to prioritise integrations with other software platforms. If you would like a more advanced UI, you can either build your own and have it call the Azure Function that is deployed by this accelerator, or look at other accelerators that may offer more narrow and specialized solutions for specific use cases or types of data.

**Can I use existing Azure resources?**

Yes - you'll need to modify the Bicep templates to refer to existing resources instead of creating new ones. See [here](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/existing-resource) for more info.

**How can I integrate with other triggers?**

- Azure Functions provides many different input & output bindings out of the box, including HTTP, Blob, CosmosDB, Event Grid and more. See [here](https://learn.microsoft.com/en-us/azure/azure-functions/functions-triggers-bindings?tabs=isolated-process%2Cpython-v2&pivots=programming-language-python) for examples showing how you can use these in your pipelines.

## Deployment

### Pricing considerations

This solution accelerator deploys multiple resources. Evaluate the cost of each component prior to deployment.

The following are links to the pricing details for some of the resources:

- [Azure OpenAI service pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/).
- [Azure AI Document Intelligence pricing](https://azure.microsoft.com/pricing/details/ai-document-intelligence/)
- [Azure AI Speech pricing](https://azure.microsoft.com/en-au/pricing/details/cognitive-services/speech-services/)
- [Azure Functions pricing](https://azure.microsoft.com/pricing/details/functions/)
- [Azure Web App Pricing](https://azure.microsoft.com/pricing/details/app-service/linux/)
- [Azure Blob Storage pricing](https://azure.microsoft.com/pricing/details/storage/blobs/)
- [Azure Key Vault pricing](https://azure.microsoft.com/en-us/pricing/details/key-vault/)
- [Azure Monitor pricing](https://azure.microsoft.com/en-us/pricing/details/monitor/)

### Deploying to Azure with `azd`

All instructions are written for unix-based systems (Linux/MacOS). While Windows instructions are coming soon, you can use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) to execute the following commands from the Linux command line.

### Prerequisites

- To use this solution accelerator, you will need access to an Azure subscription with permission to create resource groups and resources. You can [create an Azure account here for free](https://azure.microsoft.com/free/).

To customize and develop the app locally, you will need to install the following:

- A base Python 3.11 environment. You can use a package manager like `conda`, `venv` or `virtualenv` to create the environment. Once installed, make sure to have the environment activated when you start running the steps below, as it will be used as the base for isolated environments for the demo and function app.
- Clone this repository: `git clone https://github.com/azure/multimodal-ai-llm-processing-accelerator.git`
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=v4%2Cmacos%2Ccsharp%2Cportal%2Cbash#install-the-azure-functions-core-tools)

#### Deploying for the first time

Execute the following command, if you don't have any pre-existing Azure services and want to start from a fresh deployment.

1. Run `azd auth login`
1. Review the default parameters in `infra/main.bicepparam` and update as required.
1. Run `azd up` - This will provision the Azure resources and deploy the services.
    - Note: When deploying for the first time, you may receive a `ServiceUnavailable` error when attempting to deploy the apps after provisioning. If this error occurs, simple rerun `azd deploy` after 1-2 minutes.
1. After the application has been successfully deployed you will see the Function App and Web App URLs printed to the console. Open the Web App URL to interact with the demo pipelines from your browser.
It will look like the following:

![Deployed endpoints](/docs/azd-deployed-endpoints.png)

Note that the Function app is deployed on a consumption plan under the default infrastructure configuration. This means the first request after deployment or periods of inactivity will take 20-30 seconds longer while the function warms up. All requests made once the function is warm should complete in a normal timeframe.

#### Deploying again

If you've only changed the function or web app code, then you don't need to re-provision the Azure resources. You can just run:

```azd deploy --all``` or ```azd deploy api``` or ```azd deploy webapp```

If you've changed the infrastructure files (`infra` folder or `azure.yaml`), then you'll need to re-provision the Azure resources and redeploy the services. You can do that by running:

```azd up```

#### Clean up

To clean up all the resources created by this sample:

1. Remove any model deployments within the AOAI resource. If not removed, these may the resource cleanup to fail.
1. Run `azd down --purge`. This will permanently delete the resource group and all resources.

### Running the solution locally

#### Prerequisite - Deploy Azure Resources

To run the solution locally, you will need to create the necessary resources for all Azure AI service calls (e.g. Azure Document Intelligence, Azure OpenAI etc). Set these up within Azure before you start the next steps.

#### Function app local instructions

The `function_app` folder contains the backend Azure Functions App. By default, it includes a number of Azure Function Python Blueprints that showcase different types of processing pipelines.

- Open a new terminal window with a Python 3.11 environment active
- Navigate to the function app directory: `cd function_app`
- Make a local settings file to populate the necessary app settings and environment variables. You can automatically import the same app settings as the Azure deployment by following [these instructions](https://learn.microsoft.com/en-us/azure/azure-functions/functions-develop-local#synchronize-settings), and then compare the values to the the sample file: `sample_local.settings.json`. More info on local function development can be found [here](https://learn.microsoft.com/en-us/azure/azure-functions/functions-develop-local).
- Open the newly created `local.settings.json` file and populate the values for all environment variables (these are usually capitalized, e.g. `DOC_INTEL_API_KEY` or `AOAI_ENDPOINT`). You may need to go setup new resources within Azure.
- Run the environment setup script to setup a new virtual Python environment with all required dependencies for running the example: `sh setup_env.sh`.
- Activate the new environment: `source .venv/bin/activate`
- Start the function server: `func start`

Once the local web server is up and running, you can either run the demo app, or open a new terminal window and send a test request:
`sh send_req_summarize_text.sh`

#### Demo app local instructions

the `demo_app` folder contains the code for the demo web application. This application is built with the `gradio` Python web framework, and is meant for demos and testing (not production usage). The application connects to the Azure Functions server for all processing, automatically selecting the correct endpoint based on environment variables (which are set during deployment). If the server is run locally without any environment variables set, it will connect to the Function Server on `http://localhost:7071/`, otherwise it will use the `FUNCTION_HOST` and `FUNCTION_KEY` environment variables to connect to the Azure Function deployed within Azure.

- Open a new terminal window with a Python 3.11 environment active.
- Navigate to the demo app directory: `cd demo_app`
- Create a new Python 3.11 environment. To do this you can run:
  - `conda create -n mm_ai_llm_processing_demo_app python=3.11 --no-default-packages && conda activate mm_ai_llm_processing_demo_app`
  - `python -m venv .venv && source .venv/bin/activate`
- Install the python dependencies for the demo app: `pip install -r requirements.txt`
- Make a copy of the sample environment variables file, and review the values: `cp .env.sample .env`
- Start the gradio server: `gradio demo_app.py`. By default, the application will launch in auto-reload mode, automatically reloading whenever `demo_app.py` is changed.
- Open the web app in the browser: `https://localhost:8000`

# Credits

- [Michael Tremeer](https://github.com/michaeltremeer)

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
