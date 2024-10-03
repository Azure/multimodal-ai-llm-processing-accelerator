@description('The name of the Function App. This will become part of the URL (e.g. https://{functionAppName}.azurewebsites.net) and must be unique across Azure.')
param functionAppName string = 'ai-llm-processing-func'

@description('Whether to use a premium or consumption SKU for the function\'s app service plan. Premium plans are recommended for production workloads, especially non-HTTP-triggered functions.')
param functionAppUsePremiumSku bool = false

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

@description('An additional identity ID to assign storage & CosmosDB access roles to. You can use this to give storage access to your developer identity.')
param additionalRoleAssignmentIdentityIds array = []

@description('The name of the default Storage account. This should be only lowercase letters and numbers. When deployed, a unique suffix will be appended to the name.')
param storageAccountName string = 'llmprocstorage'

@description('The name of the default blob storage containers to be created')
param blobContainerNames array = ['blob-form-to-cosmosdb-blobs']

@description('The name of the default CosmosDB database')
param cosmosDbDatabaseName string = 'default'

@description('The name of the default CosmosDB containers to be created')
param cosmosDbContainerNames array = ['blob-form-to-cosmosdb-container']

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
param cosmosDbConnectionStringSecretName string = 'cosmosdb-connection-string'
param storageConnectionStringSecretName string = 'storage-connection-string'
param aoaiKeyKvSecretName string = 'aoai-api-key'
param docIntelKeyKvSecretName string = 'doc-intel-api-key'
param speechKeyKvSecretName string = 'speech-api-key'
param funcAppKeyKvSecretName string = 'func-api-key'
param appInsightsInstrumentationKeyKvSecretName string = 'appins-instrumentation-key'

var functionAppSkuProperties = functionAppUsePremiumSku
  ? {
      name: 'P0v3'
      tier: 'Premium0V3'
      size: 'P0v3'
      family: 'Pv3'
      capacity: 1
    }
  : {
      name: 'Y1'
      tier: 'Dynamic'
      size: 'Y1'
      family: 'Y'
      capacity: 0
    }

var deployOpenAILLMModel = (openAILLMDeploymentCapacity > 0)
var deployOpenAIWhisperModel = (openAIWhisperDeploymentCapacity > 0)

var functionAppTokenName = appendUniqueUrlSuffix
  ? toLower('${functionAppName}-${resourceToken}')
  : toLower(functionAppName)
var webAppTokenName = appendUniqueUrlSuffix ? toLower('${webAppName}-${resourceToken}') : toLower(webAppName)
var storageAccountTokenName = toLower('${storageAccountName}${resourceToken}')
var cosmosDbAccountTokenName = toLower('${resourcePrefix}-cosmosdb-${resourceToken}')
var functionAppPlanTokenName = toLower('${functionAppName}-plan-${resourceToken}')
var webAppPlanTokenName = toLower('${webAppName}-plan-${resourceToken}')
var openAITokenName = toLower('${resourcePrefix}-aoai-${openAILocation}-${resourceToken}')
var openAILLMDeploymentName = toLower('${openAILLMModel}-${openAILLMModelVersion}-${openAILLMDeploymentSku}')
var openAIWhisperDeploymentName = toLower('${openAIWhisperModel}-${openAIWhisperModelVersion}-${openAIWhisperDeploymentSku}')
var docIntelTokenName = toLower('${resourcePrefix}-doc-intel-${resourceToken}')
var speechTokenName = toLower('${resourcePrefix}-speech-${speechLocation}-${resourceToken}')
var logAnalyticsTokenName = toLower('${resourcePrefix}-func-la-${resourceToken}')
var appInsightsTokenName = toLower('${resourcePrefix}-func-appins-${resourceToken}')
var keyVaultName = toLower('${resourcePrefix}-kv-${resourceToken}')

//
// Storage account (the storage account is required for the Function App)
//
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountTokenName
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

// Optional blob container for storing files
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource blobStorageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = [
  for containerName in blobContainerNames: {
    name: containerName
    parent: blobServices
    properties: {
      publicAccess: 'None'
    }
  }
]

// Set list of storage role IDs - See here for more info on required roles: 
// https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference?tabs=blob&pivots=programming-language-python#connecting-to-host-storage-with-an-identity
var storageAccountContributorRoleId = '17d1049b-9a84-46fb-8f53-869881c3d3ab'
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var storageBlobDataOwnerRoleId = 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
var storageQueueDataContributorRoleId = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'

var storageRoleDefinitionIds = [
  storageAccountContributorRoleId // Storage Account Contributor
  storageBlobDataContributorRoleId // Storage Blob Data Contributor
  storageBlobDataOwnerRoleId // Storage Blob Data Owner
  storageQueueDataContributorRoleId // Storage Queue Data Contributor
]

var storageAccountConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'

//
// CosmosDB
//

// Default configuration is based on: https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/manage-with-bicep#free-tier
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2023-11-15' = {
  name: cosmosDbAccountTokenName
  location: location
  properties: {
    enableFreeTier: false
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: false
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
      }
    ]
  }
}

resource cosmosDbDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-11-15' = {
  parent: cosmosDbAccount
  name: cosmosDbDatabaseName
  properties: {
    resource: {
      id: cosmosDbDatabaseName
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-11-15' = [
  for containerName in cosmosDbContainerNames: {
    parent: cosmosDbDatabase
    name: containerName
    properties: {
      resource: {
        id: containerName
        partitionKey: {
          paths: [
            '/id'
          ]
          kind: 'Hash'
        }
        indexingPolicy: {
          indexingMode: 'consistent'
          automatic: true
        }
        defaultTtl: -1
      }
    }
  }
]

resource cosmosDbDataContributorRoleDefinition 'Microsoft.DocumentDB/databaseAccounts/sqlRoleDefinitions@2021-04-15' existing = {
  parent: cosmosDbAccount
  name: '00000000-0000-0000-0000-000000000002' // Built-in Data Contributor Role
}

var cosmosDbConnectionString = cosmosDbAccount.listConnectionStrings().connectionStrings[0].connectionString

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

resource cosmosDbConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: cosmosDbConnectionStringSecretName
  parent: keyVault
  properties: {
    value: cosmosDbConnectionString
  }
}

resource storageConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: storageConnectionStringSecretName
  parent: keyVault
  properties: {
    value: storageAccountConnectionString
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
  name: funcAppKeyKvSecretName
  parent: keyVault
  properties: {
    value: listkeys('${functionApp.id}/host/default', '2022-03-01').masterKey
  }
}

resource appInsightsInstrumentationKeyKvSecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  name: appInsightsInstrumentationKeyKvSecretName
  parent: keyVault
  properties: {
    value: appInsights.properties.ConnectionString
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
  sku: functionAppSkuProperties
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
      AzureWebJobsStorage: storageAccountConnectionString // Cannot use key vault reference here
      AzureWebJobsStorage__credential: 'managedIdentity'
      AzureWebJobsStorage__serviceUri: 'https://${storageAccount.name}.blob.core.windows.net'
      AzureWebJobsStorage__queueServiceUri: 'https://${storageAccount.name}.queue.core.windows.net'
      AzureWebJobsStorage__tableServiceUri: 'https://${storageAccount.name}.table.core.windows.net'
      WEBSITE_CONTENTSHARE: toLower(functionAppTokenName)
      WEBSITE_CONTENTAZUREFILECONNECTIONSTRING: storageAccountConnectionString // Cannot use key vault reference here
      APPLICATIONINSIGHTS_CONNECTION_STRING: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${appInsightsInstrumentationKeyKvSecretName})'
      CosmosDbConnectionSetting__accountEndpoint: cosmosDbAccount.properties.documentEndpoint
      COSMOSDB_DATABASE_NAME: cosmosDbDatabaseName
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
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

// Role assignments for the Function App's managed identity
module functionAppStorageRoleAssignments 'storage-account-role-assignment.bicep' = {
  name: 'functionAppStorageRoleAssignments'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccount.name
    identityId: functionApp.identity.principalId
    roleDefintionIds: storageRoleDefinitionIds
  }
}

module functionAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = {
  name: 'functionAppCosmosDbRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    identityId: functionApp.identity.principalId
    roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
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
      FUNCTION_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${funcAppKeyKvSecretName})'
      STORAGE_ACCOUNT_ENDPOINT: storageAccount.properties.primaryEndpoints.blob
      COSMOSDB_ACCOUNT_ENDPOINT: cosmosDbAccount.properties.documentEndpoint
      COSMOSDB_DATABASE_NAME: cosmosDbDatabaseName
      WEB_APP_USE_PASSWORD_AUTH: string(webAppUsePasswordAuth)
      WEB_APP_USERNAME: webAppUsername // Demo app username, no need for key vault storage
      WEB_APP_PASSWORD: webAppPassword // Demo app password, no need for key vault storage
    }
  }
}

// Role assignments for the Web App's managed identity

module webAppStorageRoleAssignments 'storage-account-role-assignment.bicep' = {
  name: 'webAppStorageRoleAssignments'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccount.name
    identityId: webApp.identity.principalId
    roleDefintionIds: storageRoleDefinitionIds
  }
}

module webAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = {
  name: 'webAppCosmosDbRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    identityId: webApp.identity.principalId
    roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
  }
}

// Additional role assignments (if provided)

module additionalIdentityStorageRoleAssignments 'storage-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: {
    name: 'StorageRoleAssignments-${additionalRoleAssignmentIdentityId}'
    scope: resourceGroup()
    params: {
      storageAccountName: storageAccount.name
      identityId: additionalRoleAssignmentIdentityId
      roleDefintionIds: storageRoleDefinitionIds
    }
  }
]
module additionalIdentityCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: {
    name: 'CosmosDbRoleAssignment-${additionalRoleAssignmentIdentityId}'
    scope: resourceGroup()
    params: {
      accountName: cosmosDbAccount.name
      identityId: additionalRoleAssignmentIdentityId
      roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
    }
  }
]

output FunctionAppUrl string = functionApp.properties.defaultHostName
output webAppUrl string = webApp.properties.defaultHostName
output storageAccountBlobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output cosmosDbAccountEndpoint string = cosmosDbAccount.properties.documentEndpoint
