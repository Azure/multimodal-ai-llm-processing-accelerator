using 'main.bicep'

param functionAppName = 'ai-llm-processing-func'
param webAppName = 'ai-llm-processing-demo'
param appendUniqueUrlSuffix = true
param webAppUsePasswordAuth = true
param webAppUsername = 'admin'
param webAppPassword = 'password'
param resourcePrefix = 'llm-proc'
param blobStorageAccountName = 'llmprocstorage'
param openAILocation = 'eastus2'
param openAImodel = 'gpt-4o'
param openAImodelVersion = '2024-05-13'
param openAIDeploymentSku = 'GlobalStandard'
param oaiDeploymentCapacity = 30
