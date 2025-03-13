using 'main.bicep'

param tags = {}

//// Function & web apps
param appendUniqueUrlSuffix = true

// Function app
param functionAppName = 'ai-llm-processing-func'
param functionAppUsePremiumSku = true

// Web app
// If web app deployment is not required, set deployWebApp to false and remove the webapp service deployment from the azure.yaml file
param deployWebApp = true
param webAppName = 'ai-llm-processing-demo'
param webAppUsePasswordAuth = true
param webAppUsername = 'admin'
param webAppPassword = 'password'

// Other resources
param resourcePrefix = 'llm-proc'

// Optionally give access to additional identities. This allows you to 
// run the application locally while connecting to cloud services using 
// identity-based authentication. This is required to create the Content 
// Understanding analyzers using the postprovision hook.
// To get your identity ID, run the following command in the Azure CLI:
// > az ad signed-in-user show --query id -o tsv
param additionalRoleAssignmentIdentityIds = []

// Storage service options
param storageAccountName = 'llmprocstorage'

// CosmosDB
param deployCosmosDB = true

//// Cognitive services

// Speech
param deploySpeechResource = true
// Ensure your speech service location has model availability for the methods you need - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions
// 2. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create#prerequisites
param speechLocation = 'eastus'

// Document Intelligence
param deployDocIntelResource = true
// Doc Intelligence API v4.0 is only supported in some regions. To make use of the custom
// DocumentIntelligenceProcessor, make sure to select a region where v4.0 is supported. See:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/versioning/changelog-release-history
param docIntelLocation = 'eastus'

// Content Understanding
param deployContentUnderstandingMultiServicesResource = true
// Ensure your Content Understanding resource is deployed to a supported location - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/language-region-support?tabs=document#region-support
param contentUnderstandingLocation = 'westus'

// Language
param deployLanguageResource = true
// Ensure your Language resource is deployed to a region that supports all required features - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/language-service/concepts/regional-support
param languageLocation = 'eastus'

// Azure OpenAI options
param deployOpenAIResource = true
// Ensure your OpenAI service locations have model availability - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits#regional-quota-limits
param openAILocation = 'eastus2'

param openAILLMDeploymentCapacity = 30 // Set to 0 to skip deployment
param openAILLMModel = 'gpt-4o'
param openAILLMModelVersion = '2024-05-13'
param openAILLMDeploymentSku = 'Standard'

param openAIWhisperDeploymentCapacity = 1 // Set to 0 to skip deployment 
param openAIWhisperModel = 'whisper'
param openAIWhisperModelVersion = '001'
param openAIWhisperDeploymentSku = 'Standard'

// Networking configuration

// By default, all of the resources deployed in this accelerator are networked together using either Service Endpoints or Private Endpoints.
// The configuration options below allow you to select which of these options are used and to optionally allow additional access from public networks.

// 1. Private Endpoints - This deploys a private endpoint for the group of resources and ensures all traffic to the resources stays within the VNET (no data traverses the public internet).
// 2. Service Endpoints - This uses Service Endpoints & NACL/IP rules to control internal VNET traffic, while ensuring the traffic stays within the Azure backbone network.
// Note: All access rules between the deployed resources are already configured, but you can also allow public network access using the `*AllowPublicAccess` & `*ExternalIpsOrIpRanges` parameters.

// Additionally, you can optionally allow public network access to the resources using the `*AllowPublicAccess` & `*AllowedExternalIpsOrIpRanges` parameters.
// If deploying the accelerator from a local machine over the public internet, make sure to enable public network access and add your local IP address to the `*AllowedExternalIpsOrIpRanges` parameters.
// * To disable all public network access and restrict it to only those which have a route from within the VNET, set the `*AllowPublicAccess` parameter to false. 
//    - Note that this will prevent the setup of the Content Understanding schemas and the deployment of the application code unless the deployment is done from within the VNET.
// * To enable public network access from any IP address on the internet, set the `*AllowPublicAccess` parameter to true and leave the `*AllowedExternalIpsOrIpRanges` parameter empty
// * To enable public network access but restrict it to only selected IP addresses, set the '*AllowPublicAccess' to true and included the allowed IP address in the the '*AllowedExternalIpsOrIpRanges' parameters

param webAppUsePrivateEndpoint = false
param webAppAllowPublicAccess = true
param webAppAllowedExternalIpRanges = [] // Use CIDR blocks for all entries

param functionAppNetworkingType = 'ServiceEndpoint'
param functionAppAllowPublicAccess = true
param functionAppAllowedExternalIpRanges = [] // Use CIDR blocks for all entries

param backendServicesNetworkingType = 'ServiceEndpoint'
param backendServicesAllowPublicAccess = true
param backendServicesAllowedExternalIpsOrIpRanges = [] // Use CIDR blocks for IP ranges (up to /30), otherwise use specific IP addresses.

param storageServicesAndKVNetworkingType = 'ServiceEndpoint'
param storageServicesAndKVAllowPublicAccess = true
param storageServicesAndKVAllowedExternalIpsOrIpRanges = [] // Use CIDR blocks for IP ranges (up to /30), otherwise use specific IP addresses.
