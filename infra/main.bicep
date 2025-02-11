@description('The name of the Function App. This will become part of the URL (e.g. https://{functionAppName}.azurewebsites.net) and must be unique across Azure.')
param functionAppName string = 'ai-llm-processing-func'

@description('Whether to use a premium or consumption SKU for the function\'s app service plan. Premium plans are recommended for production workloads, especially non-HTTP-triggered functions.')
param functionAppUsePremiumSku bool = false

@description('The name of the Web App. This will become part of the URL (e.g. https://{webAppName}.azurewebsites.net) and must be unique across Azure.')
param webAppName string = 'ai-llm-processing-demo'

@description('Whether to use a unique URL suffix for the Function and Web Apps (preventing name clashes with other applications). If set to true, the URL will be https://{functionAppName}-{randomToken}.azurewebsites.net')
param appendUniqueUrlSuffix bool = true

@description('Whether to deploy the Web App')
param deployWebApp bool = true

@description('Whether to require a username and password when accessing the Web App')
param webAppUsePasswordAuth bool = true

@description('The username to use when accessing the Web App if webAppUsername is true')
param webAppUsername string = 'admin'

@description('The password to use when accessing the Web App if webAppPassword is true')
@secure()
param webAppPassword string

@description('The prefix to use for all resources except the function and web apps')
param resourcePrefix string = 'llm-proc'

@description('An list of additional Azure identities to assign Ai services, storage & CosmosDB access roles to. This is necessary in order to do development locally since all calls to the backend AI services are made using identity-based auth.')
param additionalRoleAssignmentIdentityIds array = []

@description('The name of the default Storage account. This should be only lowercase letters and numbers. When deployed, a unique suffix will be appended to the name.')
param storageAccountName string = 'llmprocstorage'

@description('The name of the default blob storage containers to be created')
param blobContainerNames array = ['blob-form-to-cosmosdb-blobs', 'content-understanding-blobs']

@description('The name of the default CosmosDB database')
param cosmosDbDatabaseName string = 'default'

@description('The name of the default CosmosDB containers to be created')
param cosmosDbContainerNames array = ['blob-form-to-cosmosdb-container']

@description('Whether to deploy the Content Understanding resource (a multi-service AI resource)')
param deployContentUnderstandingMultiServicesResource bool = true

@description('The location of the Azure AI services resource to be used for Azure AI Content Understanding (this will be a multi-service resource). This should be in a location supported by the service (see https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/language-region-support?tabs=document#region-support)')
param contentUnderstandingLocation string = 'westus'

@description('Whether to deploy the Document Intelligence resource (a single-service AI resource)')
param deployDocIntelResource bool = true

@description('The location of the Azure Document Intelligence resource. This should be in a location where all required models are available (see https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/versioning/changelog-release-history)')
param docIntelLocation string = 'eastus'

@description('Whether to deploy the Speech resource (a single-service AI resource)')
param deploySpeechResource bool = true

@description('The location of the Azure AI Speech resource. This should be in a location where all required models are available (see https://learn.microsoft.com/en-au/azure/ai-services/speech-service/regions and https://learn.microsoft.com/en-au/azure/ai-services/speech-service/fast-transcription-create#prerequisites)')
param speechLocation string = 'eastus'

@description('Whether to deploy the Azure Language resource (a single-service AI resource)')
param deployLanguageResource bool = true

@description('The location of the Azure Language resource. This should be in a location where all required models are available.')
param languageLocation string = 'eastus'

@description('Whether to deploy the Azure OpenAI resource (a single-service AI resource)')
param deployOpenAIResource bool = true

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
param funcAppKeyKvSecretName string = 'func-api-key'
param appInsightsInstrumentationKeyKvSecretName string = 'appins-instrumentation-key'

// Set function app settings based on the deployment type. 
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
// The current version of this repo requires key-based storage account access to load the function app package.
// In a future version, containers will be used, enabling full identity-based access to the storage account.
var functionAppConsumptionSettings = ((!functionAppUsePremiumSku)
  ? { AzureWebJobsStorage: storageAccountConnectionString }
  : {})

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
var contentUnderstandingTokenName = toLower('${resourcePrefix}-content-understanding-${contentUnderstandingLocation}-${resourceToken}')
var docIntelTokenName = toLower('${resourcePrefix}-doc-intel-${docIntelLocation}-${resourceToken}')
var speechTokenName = toLower('${resourcePrefix}-speech-${speechLocation}-${resourceToken}')
var logAnalyticsTokenName = toLower('${resourcePrefix}-func-la-${resourceToken}')
var appInsightsTokenName = toLower('${resourcePrefix}-func-appins-${resourceToken}')
var keyVaultName = toLower('${resourcePrefix}-kv-${resourceToken}')
var languageTokenName = toLower('${resourcePrefix}-language-${languageLocation}-${resourceToken}')

// Define role definition IDs for Azure AI Services
var roleDefinitions = {
  openAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  speechUser: 'f2dc8367-1007-4938-bd23-fe263f013447'
  cognitiveServicesUser: 'a97b65f3-24c7-4388-baec-2e87135dc908'
  cosmosDbDataContributor: '00000000-0000-0000-0000-000000000002'
  storageAccountContributor: '17d1049b-9a84-46fb-8f53-869881c3d3ab'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  storageBlobDataOwner: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
  storageQueueDataContributor: '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
}

// Set list of storage role IDs - See here for more info on required roles: 
// https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference?tabs=blob&pivots=programming-language-python#connecting-to-host-storage-with-an-identity
var storageRoleDefinitionIds = [
  roleDefinitions.storageAccountContributor
  roleDefinitions.storageBlobDataContributor
  roleDefinitions.storageBlobDataOwner
  roleDefinitions.storageQueueDataContributor
]

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
    publicNetworkAccess: 'Enabled'
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
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
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

// Cognitive services resources

// Multi-service resource for Azure Content Understanding
resource contentUnderstanding 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployContentUnderstandingMultiServicesResource) {
  name: contentUnderstandingTokenName
  location: contentUnderstandingLocation
  kind: 'AIServices'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: contentUnderstandingTokenName
    disableLocalAuth: true
  }
  sku: {
    name: 'S0'
  }
}

module contentUnderstandingRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployContentUnderstandingMultiServicesResource) {
  name: guid(subscription().id, resourceGroup().id, contentUnderstanding.name, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: contentUnderstanding.name
    principalIds: union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure Document Intelligence 
resource docIntel 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployDocIntelResource) {
  name: docIntelTokenName
  location: docIntelLocation
  kind: 'FormRecognizer'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: docIntelTokenName
    disableLocalAuth: true
  }
  sku: {
    name: 'S0'
  }
}

module docIntelRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployDocIntelResource) {
  name: guid(subscription().id, resourceGroup().id, docIntel.name, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: docIntel.name
    principalIds: union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure AI Speech
resource speech 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deploySpeechResource) {
  name: speechTokenName
  location: speechLocation
  kind: 'SpeechServices'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: speechTokenName
    disableLocalAuth: true
  }
  sku: {
    name: 'S0'
  }
}

module speechRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deploySpeechResource) {
  name: guid(subscription().id, resourceGroup().id, speech.name, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: speech.name
    principalIds: union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
    roleDefinitionIds: [roleDefinitions.speechUser]
  }
}

// Azure AI Language
resource language 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployLanguageResource) {
  name: languageTokenName
  location: languageLocation
  kind: 'TextAnalytics'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: languageTokenName
    disableLocalAuth: true
  }
  sku: {
    name: 'S'
  }
}

module languageRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployLanguageResource) {
  name: guid(subscription().id, resourceGroup().id, language.name, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: language.name
    principalIds: union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure OpenAI
resource azureopenai 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployOpenAIResource) {
  name: openAITokenName
  location: openAILocation
  kind: 'OpenAI'
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: openAITokenName
    disableLocalAuth: true
  }
  sku: {
    name: 'S0'
  }
}

module openAIRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployOpenAIResource) {
  name: guid(subscription().id, resourceGroup().id, azureopenai.name, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: azureopenai.name
    principalIds: union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
    roleDefinitionIds: [roleDefinitions.openAiUser, roleDefinitions.cognitiveServicesUser]
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
    publicNetworkAccess: 'Enabled'
  }
}

resource keyVaultAccessPolicies 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  parent: keyVault
  name: 'add'
  properties: {
    accessPolicies: concat(
      [
        {
          tenantId: subscription().tenantId
          objectId: functionApp.identity.principalId
          permissions: {
            keys: ['get', 'list']
            secrets: ['get', 'list']
          }
        }
      ],
      deployWebApp
        ? [
            {
              tenantId: subscription().tenantId
              objectId: webApp.identity.principalId
              permissions: {
                keys: ['get', 'list']
                secrets: ['get', 'list']
              }
            }
          ]
        : []
    )
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

var optionalDeploymentFuncAppEnvVars = union(
  deployContentUnderstandingMultiServicesResource
    ? {}
    : {
        CONTENT_UNDERSTANDING_ENDPOINT: 'https://${contentUnderstandingTokenName}.services.ai.azure.com/'
      },
  deployDocIntelResource
    ? {}
    : {
        DOC_INTEL_ENDPOINT: docIntel.properties.endpoint
      },
  deployLanguageResource
    ? {}
    : {
        LANGUAGE_ENDPOINT: 'https://${languageTokenName}.cognitiveservices.azure.com/'
      },
  deploySpeechResource
    ? {}
    : {
        SPEECH_ENDPOINT: speech.properties.endpoint
      },
  deployOpenAIResource
    ? {}
    : {
        AOAI_ENDPOINT: azureopenai.properties.endpoint
      },
  deployOpenAILLMModel
    ? {}
    : {
        AOAI_LLM_DEPLOYMENT: openAILLMDeploymentName
      },
  deployOpenAIWhisperModel
    ? {}
    : {
        AOAI_WHISPER_DEPLOYMENT: openAIWhisperDeploymentName
      }
)

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
    properties: union(functionAppConsumptionSettings, optionalDeploymentFuncAppEnvVars, {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      AzureWebJobsFeatureFlags: 'EnableWorkerIndexing'
      AzureWebJobsStorage__credential: 'managedIdentity'
      AzureWebJobsStorage__serviceUri: 'https://${storageAccount.name}.blob.${environment().suffixes.storage}'
      AzureWebJobsStorage__queueServiceUri: 'https://${storageAccount.name}.queue.${environment().suffixes.storage}'
      AzureWebJobsStorage__tableServiceUri: 'https://${storageAccount.name}.table.${environment().suffixes.storage}'
      WEBSITE_CONTENTSHARE: toLower(functionAppTokenName)
      WEBSITE_CONTENTAZUREFILECONNECTIONSTRING: storageAccountConnectionString // Cannot use key vault reference here
      APPINSIGHTS_INSTRUMENTATIONKEY: appInsights.properties.InstrumentationKey
      APPLICATIONINSIGHTS_CONNECTION_STRING: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${appInsightsInstrumentationKeyKvSecretName})'
      CosmosDbConnectionSetting__accountEndpoint: cosmosDbAccount.properties.documentEndpoint
      COSMOSDB_DATABASE_NAME: cosmosDbDatabaseName
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
    })
  }
}

module functionAppStorageRoleAssignments 'storage-account-role-assignment.bicep' = {
  name: 'functionAppStorageRoleAssignments'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccount.name
    principalId: functionApp.identity.principalId
    roleDefintionIds: storageRoleDefinitionIds
  }
}

module functionAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = {
  name: 'functionAppCosmosDbRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    principalId: functionApp.identity.principalId
    roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
  }
}

// Demo app
resource webAppPlan 'Microsoft.Web/serverfarms@2020-06-01' = if (deployWebApp) {
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

resource webApp 'Microsoft.Web/sites@2022-03-01' = if (deployWebApp) {
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

module webAppStorageRoleAssignments 'storage-account-role-assignment.bicep' = if (deployWebApp) {
  name: 'webAppStorageRoleAssignments'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccount.name
    principalId: webApp.identity.principalId
    roleDefintionIds: storageRoleDefinitionIds
  }
}

module webAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = if (deployWebApp) {
  name: 'webAppCosmosDbRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    principalId: webApp.identity.principalId
    roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
  }
}

module additionalIdentityStorageRoleAssignments 'storage-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: {
    name: 'StorageRoleAssignments-${take(additionalRoleAssignmentIdentityId, 6)}'
    scope: resourceGroup()
    params: {
      storageAccountName: storageAccount.name
      principalId: additionalRoleAssignmentIdentityId
      roleDefintionIds: storageRoleDefinitionIds
    }
  }
]
module additionalIdentityCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: {
    name: 'CosmosDbRoleAssignment-${take(additionalRoleAssignmentIdentityId, 6)}'
    scope: resourceGroup()
    params: {
      accountName: cosmosDbAccount.name
      principalId: additionalRoleAssignmentIdentityId
      roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
    }
  }
]

// Add Language endpoint to outputs
output FunctionAppUrl string = functionApp.properties.defaultHostName
output webAppUrl string = deployWebApp ? webApp.properties.defaultHostName : ''
output storageAccountBlobEndpoint string = storageAccount.properties.primaryEndpoints.blob
output cosmosDbAccountEndpoint string = cosmosDbAccount.properties.documentEndpoint
output cosmosDbAccountName string = cosmosDbAccount.name
output cosmosDbDatabaseName string = cosmosDbDatabaseName
output AOAI_ENDPOINT string = deployOpenAIResource ? azureopenai.properties.endpoint : ''
output AOAI_LLM_DEPLOYMENT string = (deployOpenAIResource && deployOpenAILLMModel) ? openAILLMDeploymentName : ''
output AOAI_WHISPER_DEPLOYMENT string = (deployOpenAIResource && deployOpenAIWhisperModel)
  ? openAIWhisperDeploymentName
  : ''
output CONTENT_UNDERSTANDING_ENDPOINT string = deployContentUnderstandingMultiServicesResource
  ? 'https://${contentUnderstandingTokenName}.services.ai.azure.com/'
  : ''
output DOC_INTEL_ENDPOINT string = deployDocIntelResource ? docIntel.properties.endpoint : ''
output SPEECH_ENDPOINT string = deploySpeechResource ? speech.properties.endpoint : ''
output LANGUAGE_ENDPOINT string = deployLanguageResource
  ? 'https://${languageTokenName}.cognitiveservices.azure.com/'
  : ''
