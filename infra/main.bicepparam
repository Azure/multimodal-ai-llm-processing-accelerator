using 'main.bicep'

param functionAppName = 'ai-llm-processing-func'
param functionAppUsePremiumSku = false
param webAppName = 'ai-llm-processing-demo'
param appendUniqueUrlSuffix = true
param webAppUsePasswordAuth = true
param webAppUsername = 'admin'
param webAppPassword = 'password'
param resourcePrefix = 'llm-proc'
param blobStorageAccountName = 'llmprocstorage'
param speechLocation = 'eastus'
param openAILocation = 'eastus2'
param openAILLMDeploymentCapacity = 30
param openAILLMModel = 'gpt-4o'
param openAILLMModelVersion = '2024-05-13'
param openAILLMDeploymentSku = 'Standard'
param openAIWhisperDeploymentCapacity = 1
param openAIWhisperModel = 'whisper'
param openAIWhisperModelVersion = '001'
param openAIWhisperDeploymentSku = 'Standard'
