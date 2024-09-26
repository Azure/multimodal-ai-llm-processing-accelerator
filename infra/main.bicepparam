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

// Storage
param blobStorageAccountName = 'llmprocstorage'

// Cognitive services
param speechLocation = 'eastus'

// Azure OpenAI
param openAILocation = 'eastus2'
param openAILLMDeploymentCapacity = 30
param openAILLMModel = 'gpt-4o'
param openAILLMModelVersion = '2024-05-13'
param openAILLMDeploymentSku = 'Standard'
param openAIWhisperDeploymentCapacity = 1
param openAIWhisperModel = 'whisper'
param openAIWhisperModelVersion = '001'
param openAIWhisperDeploymentSku = 'Standard'
