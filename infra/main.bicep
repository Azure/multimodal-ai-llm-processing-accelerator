@description('The name of the Function App. This will become part of the URL (e.g. https://{functionAppName}.azurewebsites.net) and must be unique across Azure.')
param functionAppName string

@description('The name of the Web App. This will become part of the URL (e.g. https://{webAppName}.azurewebsites.net) and must be unique across Azure.')
param webAppName string

@description('Whether to use a unique URL suffix for the Function and Web Apps (preventing name clashes with other applications). If set to true, the URL will be https://{functionAppName}-{randomToken}.azurewebsites.net')
param appendUniqueUrlSuffix bool

@description('Whether to require a username and passworde when accessing the Web App')
param webAppUsePasswordAuth bool

@description('The username to use when accessing the Web App if webAppUsername is true')
param webAppUsername string

@description('The password to use when accessing the Web App if webAppPassword is true')
@secure()
param webAppPassword string

@description('The prefix to use for all resources except the function and web apps')
param resourcePrefix string

@description('The location of the OpenAI model deployment')
param openAILocation string

@description('The name of the Blob Storage account. This should be only lowercase letters and numbers')
param blobStorageAccountName string

@description('The OpenAI model to be deployed')
param openAImodel string

@description('The OpenAI model version to be deployed')
param openAImodelVersion string

@description('The OpenAI model deployment SKU')
param openAIDeploymentSku string

@description('The max TPM of the deployed OpenAI model, in thousands')
param oaiDeploymentCapacity int

param location string = resourceGroup().location
param resourceToken string = take(toLower(uniqueString(subscription().id, resourceGroup().id, location)), 5)
param tags object = {}
param apiServiceName string = 'api'
param webAppServiceName string = 'webapp'
param aoaiKeyKvSecretName string = 'aoai-api-key'
param docIntelKeyKvSecretName string = 'doc-intel-api-key'
param funcKeyKvSecretName string = 'func-api-key'

var functionAppTokenName = appendUniqueUrlSuffix
  ? toLower('${functionAppName}-${resourceToken}')
  : toLower(functionAppName)
var webAppTokenName = appendUniqueUrlSuffix ? toLower('${webAppName}-${resourceToken}') : toLower(webAppName)
var blobStorageAccountTokenName = toLower('${blobStorageAccountName}${resourceToken}')
var functionAppPlanTokenName = toLower('${functionAppName}-plan-${resourceToken}')
var webAppPlanTokenName = toLower('${webAppName}-plan-${resourceToken}')
var openAITokenName = toLower('${resourcePrefix}-aoai-${location}-${resourceToken}')
var oaiDeploymentName = toLower('${openAImodel}-${openAImodelVersion}-${openAIDeploymentSku}')
var docIntelTokenName = toLower('${resourcePrefix}-doc-intel-${resourceToken}')
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

resource llmdeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: azureopenai
  name: oaiDeploymentName
  properties: {
    model: {
      format: 'OpenAI'
      name: openAImodel
      version: openAImodelVersion
    }
  }
  sku: {
    name: openAIDeploymentSku
    capacity: oaiDeploymentCapacity
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
  kind: 'linux'
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  properties: {
    reserved: true
  }
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
      AzureWebJobsFeatureFlags: 'EnableWorkerIndexing'
      AzureWebJobsStorage: storageAccountConnectionString
      APPLICATIONINSIGHTS_CONNECTION_STRING: appInsights.properties.ConnectionString
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
      WEBSITE_CONTENTSHARE: toLower(functionAppTokenName)
      WEBSITE_CONTENTAZUREFILECONNECTIONSTRING: storageAccountConnectionString
      BLOB_STORAGE_CONNECTION_STRING: storageAccountConnectionString
      AOAI_ENDPOINT: azureopenai.properties.endpoint
      AOAI_DEPLOYMENT: oaiDeploymentName
      AOAI_API_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${aoaiKeyKvSecretName})'
      DOC_INTEL_ENDPOINT: docIntel.properties.endpoint
      DOC_INTEL_API_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${docIntelKeyKvSecretName})'
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

  resource appSettings 'config' = {
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

output storageAccountConnectionString string = storageAccountConnectionString
