@description('The name of the Function App. This will become part of the URL (e.g. https://{functionAppName}.azurewebsites.net) and must be unique across Azure.')
param functionAppName string = 'ai-llm-processing-func'

@description('The name of the Web App. This will become part of the URL (e.g. https://{webAppName}.azurewebsites.net) and must be unique across Azure.')
param webAppName string = 'ai-llm-processing-demo'

@description('Whether to use a unique URL suffix for the Function and Web Apps (preventing name clashes with other applications). If set to true, the URL will be https://{functionAppName}-{randomToken}.azurewebsites.net')
param appendUniqueUrlSuffix bool = true

@description('Whether to require a username and password when accessing the Web App')
param webAppUsePasswordAuth bool = true

@description('The username to use when accessing the Web App if webAppUsername is true')
param webAppUsername string = 'admin'

@description('The password to use when accessing the Web App if webAppPassword is true')
@secure()
param webAppPassword string

@description('The prefix to use for all resources except the function and web apps')
param resourcePrefix string = 'llm-proc'

@description('The name of the Blob Storage account. This should be only lowercase letters and numbers')
param blobStorageAccountName string = 'llmprocstorage'

@description('The location of the Azure AI Speech resource. This should be in a location where all required models are available (see https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions and https://learn.microsoft.com/en-au/azure/ai-services/speech-service/fast-transcription-create#prerequisites)')
param speechLocation string = 'eastus'

@description('The location of the OpenAI Azure resource. This should be in a location where all required models are available (see https://learn.microsoft.com/en-au/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability)')
param openAILocation string = 'eastus2'

@description('The max TPM of the deployed OpenAI LLM model, in thousands. `30` = 30k max TPM. If set to 0, the OpenAI LLM model will not be deployed.')
param openAILLMDeploymentCapacity int = 30

@description('The OpenAI LLM model to be deployed')
param openAILLMModel string = 'gpt-4o'

@description('The OpenAI LLM model version to be deployed')
param openAILLMModelVersion string = '2024-05-13'

@description('The OpenAI LLM model deployment SKU')
param openAILLMDeploymentSku string = 'Standard'

@description('The max TPM of the deployed OpenAI Whisper model, in thousands. `3` = 3k max TPM. If set to 0, the Whisper model will not be deployed.')
param openAIWhisperDeploymentCapacity int = 3

@description('The OpenAI Whisper model to be deployed')
param openAIWhisperModel string = 'whisper'

@description('The OpenAI Whisper model version to be deployed')
param openAIWhisperModelVersion string = '001'

@description('The OpenAI LLM model deployment SKU')
param openAIWhisperDeploymentSku string = 'Standard'

param location string = resourceGroup().location
param resourceToken string = take(toLower(uniqueString(subscription().id, resourceGroup().id, location)), 5)
param tags object = {}
param apiServiceName string = 'api'
param webAppServiceName string = 'webapp'
param aoaiKeyKvSecretName string = 'aoai-api-key'
param docIntelKeyKvSecretName string = 'doc-intel-api-key'
param speechKeyKvSecretName string = 'speech-api-key'
param funcKeyKvSecretName string = 'func-api-key'

var deployOpenAILLMModel = (openAILLMDeploymentCapacity > 0)
var deployOpenAIWhisperModel = (openAIWhisperDeploymentCapacity > 0)

var functionAppTokenName = appendUniqueUrlSuffix
  ? toLower('${functionAppName}-${resourceToken}')
  : toLower(functionAppName)
var webAppTokenName = appendUniqueUrlSuffix ? toLower('${webAppName}-${resourceToken}') : toLower(webAppName)
var blobStorageAccountTokenName = toLower('${blobStorageAccountName}${resourceToken}')
var functionAppPlanTokenName = toLower('${functionAppName}-plan-${resourceToken}')
var webAppPlanTokenName = toLower('${webAppName}-plan-${resourceToken}')
var openAITokenName = toLower('${resourcePrefix}-aoai-${location}-${resourceToken}')
var openAILLMDeploymentName = (deployOpenAILLMModel
  ? toLower('${openAILLMModel}-${openAILLMModelVersion}-${openAILLMDeploymentSku}')
  : 'LLM_IS_NOT_DEPLOYED')
var openAIWhisperDeploymentName = (deployOpenAIWhisperModel
  ? toLower('${openAIWhisperModel}-${openAIWhisperModelVersion}-${openAIWhisperDeploymentSku}')
  : 'WHISPER_IS_NOT_DEPLOYED')
var docIntelTokenName = toLower('${resourcePrefix}-doc-intel-${resourceToken}')
var speechTokenName = toLower('${resourcePrefix}-speech-${resourceToken}')
var logAnalyticsTokenName = toLower('${resourcePrefix}-func-la-${resourceToken}')
var appInsightsTokenName = toLower('${resourcePrefix}-func-appins-${resourceToken}')
var keyVaultName = toLower('${resourcePrefix}-kv-${resourceToken}')

//
// Blob Storage
//
resource blobStorageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: blobStorageAccountTokenName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    defaultToOAuthAuthentication: true
    allowBlobPublicAccess: false
  }
}

var storageAccountConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${blobStorageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${blobStorageAccount.listKeys().keys[0].value}'

// Cognitive services resources

resource docIntel 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: docIntelTokenName
  location: location
  kind: 'FormRecognizer'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: docIntelTokenName
  }
  sku: {
    name: 'S0'
  }
}

resource speech 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: speechTokenName
  location: speechLocation
  kind: 'SpeechServices'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: speechTokenName
  }
  sku: {
    name: 'S0'
  }
}

resource azureopenai 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAITokenName
  location: openAILocation
  kind: 'OpenAI'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: openAITokenName
  }
  sku: {
    name: 'S0'
  }
}

resource llmdeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (deployOpenAILLMModel) {
  parent: azureopenai
  name: openAILLMDeploymentName
  properties: {
    model: {
      format: 'OpenAI'
      name: openAILLMModel
      version: openAILLMModelVersion
    }
  }
  sku: {
    name: openAILLMDeploymentSku
    capacity: openAILLMDeploymentCapacity
  }
}

resource whisperdeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (deployOpenAIWhisperModel) {
  parent: azureopenai
  dependsOn: [llmdeployment] // Ensure one model deployment at a time
  name: openAIWhisperDeploymentName
  properties: {
    model: {
      format: 'OpenAI'
      name: openAIWhisperModel
      version: openAIWhisperModelVersion
    }
  }
  sku: {
    name: openAIWhisperDeploymentSku
    capacity: openAIWhisperDeploymentCapacity
  }
}

// Key Vault for storing API keys
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      name: 'standard'
      family: 'A'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
  }
}

resource keyVaultAccessPolicy 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  parent: keyVault
  name: 'add'
  properties: {
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: functionApp.identity.principalId
        permissions: {
          keys: ['get', 'list']
          secrets: ['get', 'list']
        }
      }
      {
        tenantId: subscription().tenantId
        objectId: webApp.identity.principalId
        permissions: {
          keys: ['get', 'list']
          secrets: ['get', 'list']
        }
      }
    ]
  }
}

resource openAIKvSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: aoaiKeyKvSecretName
  parent: keyVault
  properties: {
    value: azureopenai.listKeys().key1
  }
}

resource docIntelKvSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: docIntelKeyKvSecretName
  parent: keyVault
  properties: {
    value: docIntel.listKeys().key1
  }
}

resource SpeechKvSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: speechKeyKvSecretName
  parent: keyVault
  properties: {
    value: speech.listKeys().key1
  }
}

resource funcAppKvSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: funcKeyKvSecretName
  parent: keyVault
  properties: {
    value: listkeys('${functionApp.id}/host/default', '2022-03-01').masterKey
  }
}

/// Function App
// Create a new Log Analytics workspace to back the Azure Application Insights instance
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2021-06-01' = {
  name: logAnalyticsTokenName
  location: location
}

// Application Insights instance
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsTokenName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource functionAppPlan 'Microsoft.Web/serverfarms@2020-06-01' = {
  name: functionAppPlanTokenName
  location: location
  properties: {
    reserved: true
  }
  sku: {
    name: 'P0v3'
    tier: 'Premium0V3'
    size: 'P0v3'
    family: 'Pv3'
    capacity: 1
  }
  kind: 'linux'
}

resource functionApp 'Microsoft.Web/sites@2020-06-01' = {
  name: functionAppTokenName
  kind: 'functionapp,linux'
  location: location
  tags: union(tags, { 'azd-service-name': apiServiceName })
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    httpsOnly: true
    serverFarmId: functionAppPlan.id
    clientAffinityEnabled: true
    siteConfig: {
      pythonVersion: '3.11'
      linuxFxVersion: 'python|3.11'
      cors: {
        allowedOrigins: [
          '*'
        ]
      }
    }
  }
  resource functionSettings 'config@2022-09-01' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      AzureWebJobsFeatureFlags: 'EnableWorkerIndexing'
      AzureWebJobsStorage: storageAccountConnectionString
      APPLICATIONINSIGHTS_CONNECTION_STRING: appInsights.properties.ConnectionString
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
      WEBSITE_CONTENTSHARE: toLower(functionAppTokenName)
      WEBSITE_CONTENTAZUREFILECONNECTIONSTRING: storageAccountConnectionString
      BLOB_STORAGE_CONNECTION_STRING: storageAccountConnectionString
      AOAI_ENDPOINT: azureopenai.properties.endpoint
      AOAI_LLM_DEPLOYMENT: openAILLMDeploymentName
      AOAI_WHISPER_DEPLOYMENT: openAIWhisperDeploymentName
      AOAI_API_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${aoaiKeyKvSecretName})'
      DOC_INTEL_ENDPOINT: docIntel.properties.endpoint
      DOC_INTEL_API_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${docIntelKeyKvSecretName})'
      SPEECH_REGION: speech.location
      SPEECH_ENDPOINT: speech.properties.endpoint
      SPEECH_API_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${speechKeyKvSecretName})'
    }
  }
}

// Demo app
resource webAppPlan 'Microsoft.Web/serverfarms@2020-06-01' = {
  name: webAppPlanTokenName
  location: location
  tags: tags
  properties: {
    reserved: true
  }
  sku: {
    name: 'P0v3'
    tier: 'Premium0V3'
    size: 'P0v3'
    family: 'Pv3'
    capacity: 1
  }
  kind: 'linux'
  dependsOn: [functionAppPlan] // Consumption plan must be deployed before premium plan
}

resource webApp 'Microsoft.Web/sites@2022-03-01' = {
  name: webAppTokenName
  location: location
  tags: union(tags, { 'azd-service-name': webAppServiceName })
  kind: 'app,linux'
  properties: {
    clientAffinityEnabled: true // If app plan capacity > 1, set this to True to ensure gradio works correctly.
    serverFarmId: webAppPlan.id
    siteConfig: {
      alwaysOn: true
      linuxFxVersion: 'python|3.11'
      ftpsState: 'Disabled'
      appCommandLine: 'python demo_app.py'
      minTlsVersion: '1.2'
    }
    httpsOnly: true
  }
  identity: {
    type: 'SystemAssigned'
  }

  resource appSettings 'config@2022-09-01' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      FUNCTION_HOSTNAME: 'https://${functionApp.properties.defaultHostName}'
      FUNCTION_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${funcKeyKvSecretName})'
      WEB_APP_USE_PASSWORD_AUTH: string(webAppUsePasswordAuth)
      WEB_APP_USERNAME: webAppUsername // Demo app username, no need for key vault storage
      WEB_APP_PASSWORD: webAppPassword // Demo app password, no need for key vault storage
    }
  }
}
