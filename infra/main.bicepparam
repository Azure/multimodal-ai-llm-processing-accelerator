using 'main.bicep'

// Function & web apps
param appendUniqueUrlSuffix = true

param functionAppName = 'ai-llm-processing-func'
param functionAppUsePremiumSku = true

param webAppName = 'ai-llm-processing-demo'
param webAppUsePasswordAuth = true
param webAppUsername = 'admin'
param webAppPassword = 'password'

// Other resources
param resourcePrefix = 'llm-proc'

// Optionally give access to additional identities. This allows you to 
// run the application locally while connecting to cloud services using 
// identity-based authentication.
// To get your identity ID, run the following command in the Azure CLI:
// > az ad signed-in-user show --query id -o tsv
param additionalRoleAssignmentIdentityIds = []

// Storage
param storageAccountName = 'llmprocstorage'

// Cognitive services
// Ensure your speech service location has model availability for the methods you need - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions
// 2. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create#prerequisites
param speechLocation = 'eastus'

// Azure OpenAI
// Ensure your OpenAI service locations have model availability - see:
// 1. https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits#regional-quota-limits
param openAILocation = 'eastus2'
param openAILLMDeploymentCapacity = 30
param openAILLMModel = 'gpt-4o'
param openAILLMModelVersion = '2024-05-13'
param openAILLMDeploymentSku = 'Standard'
param openAIWhisperDeploymentCapacity = 1
param openAIWhisperModel = 'whisper'
param openAIWhisperModelVersion = '001'
param openAIWhisperDeploymentSku = 'Standard'
