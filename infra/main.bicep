@description('The name of the Function App. This will become part of the URL (e.g. https://{functionAppName}.azurewebsites.net) and must be unique across Azure.')
param functionAppName string = 'ai-llm-processing-func'

@description('Whether to use a premium or consumption SKU for the function\'s app service plan. Premium plans are recommended for production workloads, especially non-HTTP-triggered functions.')
param functionAppUsePremiumSku bool = false

@description('The name of the Web App. This will become part of the URL (e.g. https://{webAppName}.azurewebsites.net) and must be unique across Azure.')
param webAppName string = 'ai-llm-processing-demo'

@description('Whether to use a unique URL suffix for the Function and Web Apps (preventing name clashes with other applications). If set to true, the URL will be https://{functionAppName}-{randomToken}.azurewebsites.net')
param appendUniqueUrlSuffix bool = true

@description('Whether to deploy the Web App. If web app deployment is not required, set deployWebApp to false and remove the webapp service deployment from the azure.yaml file')
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

@description('Whether to deploy the CosmosDB resource. This is required if using CosmosDB as an input data source or when writing records to an output CosmosDB container.')
param deployCosmosDB bool = false

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

@description('Whether to deploy the Azure OpenAI resource (a single-service AI resource). If set to false, all OpenAI-related model deployments will be skipped.')
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

// Networking parameters
@description('The address space for the virtual network (CIDR notation)')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('The address space for the backend services subnet (CIDR notation)')
param backendServicesSubnetPrefix string = '10.0.1.0/24'

@description('The address space for the function app subnet (CIDR notation)')
param functionAppSubnetPrefix string = '10.0.2.0/24'

@description('The address space for the shared storage & key vaultsubnet (CIDR notation)')
param storageServicesAndKVSubnetPrefix string = '10.0.3.0/24'

@description('The address space for the web app subnet (CIDR notation)')
param webAppSubnetPrefix string = '10.0.4.0/24'

@description('The address space for the function app private endpoint subnet (CIDR notation)')
param functionAppPrivateEndpointSubnetPrefix string = '10.0.5.0/24'

@description('The address space for the web app private endpoint subnet (CIDR notation)')
param webAppPrivateEndpointSubnetPrefix string = '10.0.6.0/24'

// Add networking type parameters
@description('Whether to deploy a private endpoint as the network endpoint for the web app (if the web app deployment is enabled).')
param webAppUsePrivateEndpoint bool = false

@description('The type of network endpoint to use for the Function app. ServiceEndpoint enables public access to the deployed service but limits access to the IP addresses specified in the functionAppExternalIpsOrIpRanges parameter. PrivateEndpoint prevents public network access and deploys Private Endpoints into the function app subnet, restricting access to clients that have access to that subnet.')
@allowed([
  'ServiceEndpoint'
  'PrivateEndpoint'
])
param functionAppNetworkingType string

@description('The type of network endpoint to use for backend AI services. ServiceEndpoint enables public access to the deployed services but limits access to the IP addresses specified in the backendServicesExternalIpsOrIpRanges parameter. PrivateEndpoint prevents public network access and deploys Private Endpoints into the backend subnet, restricting access to clients that have access to that subnet.')
@allowed([
  'ServiceEndpoint'
  'PrivateEndpoint'
])
param backendServicesNetworkingType string

@description('The type of network endpoint to use for storage services. ServiceEndpoint enables public access to the deployed services but limits access to the IP addresses specified in the storageServicesAndKVExternalIpsOrIpRanges parameter. PrivateEndpoint prevents public network access and deploys Private Endpoints into the storage subnet, restricting access to clients that have access to that subnet.')
@allowed([
  'ServiceEndpoint'
  'PrivateEndpoint'
])
param storageServicesAndKVNetworkingType string

// External developer access rules
@description('Whether to allow public access to the web app (only applied when not using a private endpoint)')
param webAppAllowPublicAccess bool = true

@description('When not using a private endpoint and where webAppAllowPublicAccess is true, defines the CIDR ranges allowed for external access to the web app')
param webAppAllowedExternalIpRanges array = []

@description('Whether to allow public access to the function app (only applied when not using a private endpoint)')
param functionAppAllowPublicAccess bool = true

@description('When using a service endpoint and where functionAppAllowPublicAccess is true, defines the CIDR ranges allowed for external access to the function app')
param functionAppAllowedExternalIpRanges array = []

@description('Whether to allow public access to the backend AI services (only applied when not using a private endpoint)')
param backendServicesAllowPublicAccess bool = true

@description('When using service endpoints and where backendServicesAllowPublicAccess is true, defines the IP addresses and/or CIDR ranges allowed for external access to the backend AI services')
param backendServicesAllowedExternalIpsOrIpRanges array = []

@description('Whether to allow public access to the storage & Key Vault services (only applied when not using a private endpoint)')
param storageServicesAndKVAllowPublicAccess bool = true

@description('When using service endpoints and where storageServicesAndKVAllowPublicAccess is true, defines the IP addresses and/or CIDR ranges allowed for external access to the storage & Key Vault services')
param storageServicesAndKVAllowedExternalIpsOrIpRanges array = []

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

// Define role definition IDs for Azure AI Services & Storage
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

// Create a unique string for the list of additional identity assign to assign roles - this ensures any changes to the list will trigger a redeploy of the role assignments module
var roleAssignmentIdentityIds = union([functionApp.identity.principalId], additionalRoleAssignmentIdentityIds)
var additionalRoleAssignmentIdsStr = join(additionalRoleAssignmentIdentityIds, ',')

// Network configuration

var webAppSubnetName = 'web-app-subnet'
var webAppPrivateEndpointSubnetName = 'web-app-pe-subnet'
var functionAppSubnetName = 'function-app-subnet'
var functionAppPrivateEndpointSubnetName = 'function-app-pe-subnet'
var backendServicesSubnetName = 'backend-services-subnet'
var storageServicesAndKVSubnetName = 'storage-services-subnet'

var privateCognitiveServicesDnsZoneName = 'privatelink.cognitiveservices.azure.com'
var privateOpenAIDnsZoneName = 'privatelink.openai.azure.com'
var privateContentUnderstandingDnsZoneName = 'privatelink.services.ai.azure.com'
var privateCosmosDbDnsZoneName = 'privatelink.documents.azure.com'
var privateStorageFileDnsZoneName = 'privatelink.file.${environment().suffixes.storage}'
var privateStorageTableDnsZoneName = 'privatelink.table.${environment().suffixes.storage}'
var privateStorageBlobDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'
var privateStorageQueueDnsZoneName = 'privatelink.queue.${environment().suffixes.storage}'
var privateAppDnsZoneName = 'privatelink.azurewebsites.net'
var privateKeyVaultDnsZoneName = 'privatelink.vaultcore.azure.net'

// Virtual Network

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${resourcePrefix}-vnet-${resourceToken}'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: backendServicesSubnetName
        properties: {
          addressPrefix: backendServicesSubnetPrefix
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: backendServicesNsg.id
          }
        }
      }
      {
        name: functionAppSubnetName
        properties: {
          addressPrefix: functionAppSubnetPrefix
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          delegations: [
            {
              name: 'Microsoft.Web.serverFarms'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
          networkSecurityGroup: {
            id: functionAppNsg.id
          }
          // Only deploy relevant service endpoints based on networking type
          serviceEndpoints: union(
            storageServicesAndKVNetworkingType == 'ServiceEndpoint'
              ? [
                  {
                    service: 'Microsoft.Storage'
                    locations: ['*']
                  }
                  {
                    service: 'Microsoft.AzureCosmosDB'
                    locations: ['*']
                  }
                ]
              : [],
            backendServicesNetworkingType == 'ServiceEndpoint'
              ? [
                  {
                    service: 'Microsoft.CognitiveServices'
                    locations: ['*']
                  }
                ]
              : []
          )
        }
      }
      {
        name: storageServicesAndKVSubnetName
        properties: {
          addressPrefix: storageServicesAndKVSubnetPrefix
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: storageServicesAndKVNsg.id
          }
        }
      }
      {
        name: webAppSubnetName
        properties: {
          addressPrefix: webAppSubnetPrefix
          delegations: [
            {
              name: 'Microsoft.Web.serverFarms'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: webAppNsg.id
          }
          // Only deploy relevant service endpoints based on networking type
          serviceEndpoints: union(
            functionAppNetworkingType == 'ServiceEndpoint'
              ? [
                  {
                    service: 'Microsoft.Web'
                    locations: ['*']
                  }
                ]
              : [],
            storageServicesAndKVNetworkingType == 'ServiceEndpoint'
              ? [
                  {
                    service: 'Microsoft.KeyVault'
                    locations: ['*']
                  }
                  {
                    service: 'Microsoft.Storage'
                    locations: ['*']
                  }
                  {
                    service: 'Microsoft.AzureCosmosDB'
                    locations: ['*']
                  }
                ]
              : []
          )
        }
      }
      {
        name: functionAppPrivateEndpointSubnetName
        properties: {
          addressPrefix: functionAppPrivateEndpointSubnetPrefix
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: functionAppPrivateEndpointNsg.id
          }
        }
      }
      {
        name: webAppPrivateEndpointSubnetName
        properties: {
          addressPrefix: webAppPrivateEndpointSubnetPrefix
          privateEndpointNetworkPolicies: 'Enabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: webAppPrivateEndpointNsg.id
          }
        }
      }
    ]
  }
}

// NSG for backend services - most restricted
resource backendServicesNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-backend-services-nsg-${resourceToken}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowInternalAccess'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefixes: [
            functionAppSubnetPrefix // Only allow access from function app subnet
          ]
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          priority: 4096
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

// NSG for Function App - allows web app and developer access
resource functionAppNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-function-app-nsg-${resourceToken}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowInternalAccess'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefixes: [
            webAppSubnetPrefix // Allow access from web app subnet
          ]
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          priority: 4096
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

// NSG for shared storage & key vault - allows access from both function app and web app
resource storageServicesAndKVNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-storage-nsg-${resourceToken}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowInternalAccess'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefixes: [
            functionAppSubnetPrefix // Allow access from function app subnet
            webAppSubnetPrefix // Allow access from web app subnet
          ]
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          priority: 4096
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

// NSG for web app - no internal routes necessary as it is only used for public access
resource webAppNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-web-app-nsg-${resourceToken}'
  location: location
}

// Create NSG for function app private endpoint subnet - only allow access from web app
resource functionAppPrivateEndpointNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-function-app-pe-nsg-${resourceToken}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowFromWebAppSubnet'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: webAppSubnetPrefix
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          priority: 4096
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

// Create NSG for web app private endpoint subnet - allow access from the whole VNET
resource webAppPrivateEndpointNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${resourcePrefix}-web-app-pe-nsg-${resourceToken}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowAllInbound'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: vnetAddressPrefix
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
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
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
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
    publicNetworkAccess: storageServicesAndKVAllowPublicAccess ? 'Enabled' : 'Disabled'
    networkAcls: {
      bypass: 'Logging,Metrics' // Allow logging and metrics to bypass network rules
      defaultAction: 'Deny'
      // Only include VNET rules when using service endpoints
      virtualNetworkRules: storageServicesAndKVNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              action: 'Allow'
            }
            {
              id: '${vnet.id}/subnets/${webAppSubnetName}'
              action: 'Allow'
            }
          ]
        : []
      // Only include IP rules when using service endpoints and public access rules are enabled
      ipRules: (storageServicesAndKVAllowPublicAccess && !empty(storageServicesAndKVAllowedExternalIpsOrIpRanges))
        ? map(storageServicesAndKVAllowedExternalIpsOrIpRanges, ip => {
            value: ip
            action: 'Allow'
          })
        : []
    }
  }
}

var storageAccountBlobUri = 'https://${storageAccount.name}.blob.${environment().suffixes.storage}'
var storageAccountQueueUri = 'https://${storageAccount.name}.queue.${environment().suffixes.storage}'
var storageAccountTableUri = 'https://${storageAccount.name}.table.${environment().suffixes.storage}'

// Optional blob container for storing files
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource blobStorageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [
  for containerName in blobContainerNames: {
    name: containerName
    parent: blobServices
    properties: {
      publicAccess: 'None'
    }
  }
]

resource privateStorageFileDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateStorageFileDnsZoneName
  location: 'global'
}

resource privateStorageBlobDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateStorageBlobDnsZoneName
  location: 'global'
}

resource privateStorageQueueDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateStorageQueueDnsZoneName
  location: 'global'
}

resource privateStorageTableDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateStorageTableDnsZoneName
  location: 'global'
}

resource privateStorageFileDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateStorageFileDnsZone
  name: '${privateStorageFileDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource privateStorageBlobDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateStorageBlobDnsZone
  name: '${privateStorageBlobDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource privateStorageTableDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateStorageTableDnsZone
  name: '${privateStorageTableDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource privateStorageQueueDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateStorageQueueDnsZone
  name: '${privateStorageQueueDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource privateEndpointStorageFile 'Microsoft.Network/privateEndpoints@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${storageAccount.name}-file-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, backendServicesSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyStorageFilePrivateLinkConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'file'
          ]
        }
      }
    ]
  }
}

resource privateEndpointStorageBlob 'Microsoft.Network/privateEndpoints@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${storageAccount.name}-blob-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, backendServicesSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyStorageBlobPrivateLinkConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'blob'
          ]
        }
      }
    ]
  }
}

resource privateEndpointStorageTable 'Microsoft.Network/privateEndpoints@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${storageAccount.name}-table-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, backendServicesSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyStorageTablePrivateLinkConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'table'
          ]
        }
      }
    ]
  }
}

resource privateEndpointStorageQueue 'Microsoft.Network/privateEndpoints@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${storageAccount.name}-queue-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, backendServicesSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyStorageQueuePrivateLinkConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'queue'
          ]
        }
      }
    ]
  }
}

resource privateEndpointStorageFilePrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateEndpointStorageFile
  name: 'filePrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'config'
        properties: {
          privateDnsZoneId: privateStorageFileDnsZone.id
        }
      }
    ]
  }
}

resource privateEndpointStorageBlobPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateEndpointStorageBlob
  name: 'blobPrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'config'
        properties: {
          privateDnsZoneId: privateStorageBlobDnsZone.id
        }
      }
    ]
  }
}

resource privateEndpointStorageTablePrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateEndpointStorageTable
  name: 'tablePrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'config'
        properties: {
          privateDnsZoneId: privateStorageTableDnsZone.id
        }
      }
    ]
  }
}

resource privateEndpointStorageQueuePrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateEndpointStorageQueue
  name: 'queuePrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'config'
        properties: {
          privateDnsZoneId: privateStorageQueueDnsZone.id
        }
      }
    ]
  }
}

var storageAccountConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'

//
// CosmosDB with flexible networking
//

// Default configuration is based on: https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/manage-with-bicep#free-tier
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2024-11-15' = if (deployCosmosDB) {
  name: cosmosDbAccountTokenName
  location: location
  properties: {
    enableFreeTier: false
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: true
    publicNetworkAccess: storageServicesAndKVAllowPublicAccess ? 'Enabled' : 'Disabled'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
      }
    ]
    networkAclBypass: 'None'
    // Only enable VNET filtering when using service endpoints
    isVirtualNetworkFilterEnabled: storageServicesAndKVNetworkingType == 'ServiceEndpoint'
    // Only include VNET rules when using service endpoints
    virtualNetworkRules: storageServicesAndKVNetworkingType == 'ServiceEndpoint'
      ? [
          {
            id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, webAppSubnetName)
            ignoreMissingVNetServiceEndpoint: false
          }
          {
            id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, functionAppSubnetName)
            ignoreMissingVNetServiceEndpoint: false
          }
        ]
      : []
    // Only include IP rules when using service endpoints and public access rules are enabled
    ipRules: (storageServicesAndKVAllowPublicAccess && !empty(storageServicesAndKVAllowedExternalIpsOrIpRanges))
      ? map(storageServicesAndKVAllowedExternalIpsOrIpRanges, ip => {
          ipAddressOrRange: ip
        })
      : []
  }
}

resource cosmosDbPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployCosmosDB && storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${cosmosDbAccount.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${storageServicesAndKVSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${cosmosDbAccount.name}-plsc'
        properties: {
          privateLinkServiceId: cosmosDbAccount.id
          groupIds: ['Sql']
        }
      }
    ]
  }

  resource cosmosDbPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'cosmosDbPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: cosmosDbPrivateDnsZone.id
          }
        }
      ]
    }
  }
}

// Add CosmosDB DNS zone for private endpoints
resource cosmosDbPrivateDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (deployCosmosDB && storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateCosmosDbDnsZoneName
  location: 'global'
}

// Add DNS zone VNet link for CosmosDB
resource cosmosDbPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (deployCosmosDB && storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: cosmosDbPrivateDnsZone
  name: 'link-to-vnet'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource cosmosDbDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-11-15' = if (deployCosmosDB) {
  parent: cosmosDbAccount
  name: cosmosDbDatabaseName
  properties: {
    resource: {
      id: cosmosDbDatabaseName
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-11-15' = [
  for containerName in cosmosDbContainerNames: if (deployCosmosDB) {
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

resource cosmosDbDataReaderRoleDefinition 'Microsoft.DocumentDB/databaseAccounts/sqlRoleDefinitions@2021-04-15' existing = {
  parent: cosmosDbAccount
  name: '00000000-0000-0000-0000-000000000001' // Built-in Data Reader Role
}

// Cognitive services resources

// Multi-service resource for Azure Content Understanding
resource contentUnderstanding 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployContentUnderstandingMultiServicesResource) {
  name: contentUnderstandingTokenName
  location: contentUnderstandingLocation
  kind: 'AIServices'
  properties: {
    publicNetworkAccess: backendServicesAllowPublicAccess ? 'Enabled' : 'Disabled'
    customSubDomainName: contentUnderstandingTokenName
    disableLocalAuth: true
    networkAcls: {
      bypass: 'None'
      defaultAction: 'Deny'
      // Only add virtual network rules if using service endpoints
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      // Only add IP rules if using service endpoints and public access rules are enabled
      ipRules: (backendServicesAllowPublicAccess && !empty(backendServicesAllowedExternalIpsOrIpRanges))
        ? map(backendServicesAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
  sku: {
    name: 'S0'
  }
}

resource contentUnderstandingPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployContentUnderstandingMultiServicesResource && backendServicesNetworkingType == 'PrivateEndpoint') {
  name: '${contentUnderstanding.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${backendServicesSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${contentUnderstanding.name}-plsc'
        properties: {
          privateLinkServiceId: contentUnderstanding.id
          groupIds: ['account']
        }
      }
    ]
  }

  resource contentUnderstandingPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'contentUnderstandingPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateContentUnderstandingDnsZone.id
          }
        }
      ]
    }
  }
}

resource privateContentUnderstandingDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint' && deployContentUnderstandingMultiServicesResource) {
  name: privateContentUnderstandingDnsZoneName
  location: 'global'
}

resource privateContentUnderstandingDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint' && deployContentUnderstandingMultiServicesResource) {
  parent: privateContentUnderstandingDnsZone
  name: '${privateContentUnderstandingDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

module contentUnderstandingRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployContentUnderstandingMultiServicesResource) {
  name: guid(contentUnderstanding.id, additionalRoleAssignmentIdsStr, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: contentUnderstanding.name
    principalIds: roleAssignmentIdentityIds
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure Document Intelligence 
resource docIntel 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployDocIntelResource) {
  name: docIntelTokenName
  location: docIntelLocation
  kind: 'FormRecognizer'
  properties: {
    publicNetworkAccess: backendServicesAllowPublicAccess ? 'Enabled' : 'Disabled'
    customSubDomainName: docIntelTokenName
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Deny'
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      ipRules: (backendServicesAllowPublicAccess && !empty(backendServicesAllowedExternalIpsOrIpRanges))
        ? map(backendServicesAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
  sku: {
    name: 'S0'
  }
}

resource privateCognitiveServicesDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint') {
  name: privateCognitiveServicesDnsZoneName
  location: 'global'
}

resource privateCognitiveServicesDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint') {
  parent: privateCognitiveServicesDnsZone
  name: '${privateCognitiveServicesDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource docIntelPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployDocIntelResource && backendServicesNetworkingType == 'PrivateEndpoint') {
  name: '${docIntel.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${backendServicesSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${docIntel.name}-plsc'
        properties: {
          privateLinkServiceId: docIntel.id
          groupIds: ['account']
        }
      }
    ]
  }

  resource docIntelPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'docIntelPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateCognitiveServicesDnsZone.id
          }
        }
      ]
    }
  }
}

module docIntelRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployDocIntelResource) {
  name: guid(docIntel.id, additionalRoleAssignmentIdsStr, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: docIntel.name
    principalIds: roleAssignmentIdentityIds
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure AI Speech
resource speech 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deploySpeechResource) {
  name: speechTokenName
  location: speechLocation
  kind: 'SpeechServices'
  properties: {
    publicNetworkAccess: backendServicesAllowPublicAccess ? 'Enabled' : 'Disabled'
    customSubDomainName: speechTokenName
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Deny'
      // Only include VNET rules when using service endpoints
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      // Only include IP rules when using service endpoints and public access rules are enabled
      ipRules: (backendServicesAllowPublicAccess && !empty(backendServicesAllowedExternalIpsOrIpRanges))
        ? map(backendServicesAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
  sku: {
    name: 'S0'
  }
}

resource speechPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deploySpeechResource && backendServicesNetworkingType == 'PrivateEndpoint') {
  name: '${speech.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${backendServicesSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${speech.name}-plsc'
        properties: {
          privateLinkServiceId: speech.id
          groupIds: ['account']
        }
      }
    ]
  }

  resource speechPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'speechPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateCognitiveServicesDnsZone.id
          }
        }
      ]
    }
  }
}

module speechRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deploySpeechResource) {
  name: guid(speech.id, additionalRoleAssignmentIdsStr, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: speech.name
    principalIds: roleAssignmentIdentityIds
    roleDefinitionIds: [roleDefinitions.speechUser]
  }
}

// Azure AI Language
resource language 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployLanguageResource) {
  name: languageTokenName
  location: languageLocation
  kind: 'TextAnalytics'
  properties: {
    publicNetworkAccess: backendServicesAllowPublicAccess ? 'Enabled' : 'Disabled'
    customSubDomainName: languageTokenName
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Deny'
      // Only include VNET rules when using service endpoints
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      // Only include IP rules when using service endpoints and public access rules are enabled
      ipRules: (backendServicesAllowPublicAccess && !empty(backendServicesAllowedExternalIpsOrIpRanges))
        ? map(backendServicesAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
  sku: {
    name: 'S'
  }
}

resource languagePrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployLanguageResource && backendServicesNetworkingType == 'PrivateEndpoint') {
  name: '${language.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${backendServicesSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${language.name}-plsc'
        properties: {
          privateLinkServiceId: language.id
          groupIds: ['account']
        }
      }
    ]
  }

  resource languagePrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'languagePrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateCognitiveServicesDnsZone.id
          }
        }
      ]
    }
  }
}

module languageRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployLanguageResource) {
  name: guid(language.id, additionalRoleAssignmentIdsStr, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: language.name
    principalIds: roleAssignmentIdentityIds
    roleDefinitionIds: [roleDefinitions.cognitiveServicesUser]
  }
}

// Azure OpenAI
resource azureopenai 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployOpenAIResource) {
  name: openAITokenName
  location: openAILocation
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    publicNetworkAccess: backendServicesAllowPublicAccess ? 'Enabled' : 'Disabled'
    customSubDomainName: openAITokenName
    disableLocalAuth: true
    networkAcls: {
      bypass: 'None'
      defaultAction: 'Deny'
      // Only include VNET rules when using service endpoints
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${functionAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      // Only include IP rules when using service endpoints and public access rules are enabled
      ipRules: (backendServicesAllowPublicAccess && !empty(backendServicesAllowedExternalIpsOrIpRanges))
        ? map(backendServicesAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
}

resource openAIPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (deployOpenAIResource && backendServicesNetworkingType == 'PrivateEndpoint') {
  name: '${azureopenai.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${backendServicesSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${azureopenai.name}-plsc'
        properties: {
          privateLinkServiceId: azureopenai.id
          groupIds: ['account']
        }
      }
    ]
  }

  resource openAIPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'openAIPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateOpenAIDnsZone.id
          }
        }
      ]
    }
  }
}

resource privateOpenAIDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint' && deployOpenAIResource) {
  name: privateOpenAIDnsZoneName
  location: 'global'
}

resource privateOpenAIDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (backendServicesNetworkingType == 'PrivateEndpoint' && deployOpenAIResource) {
  parent: privateOpenAIDnsZone
  name: '${privateOpenAIDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

module openAIRoleAssignments 'multiple-ai-services-role-assignment.bicep' = if (deployOpenAIResource) {
  name: guid(azureopenai.id, additionalRoleAssignmentIdsStr, 'AiServicesRoleAssignments')
  params: {
    aiServiceName: azureopenai.name
    principalIds: roleAssignmentIdentityIds
    roleDefinitionIds: [roleDefinitions.openAiUser, roleDefinitions.cognitiveServicesUser]
  }
}

resource llmdeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAIResource && deployOpenAILLMModel) {
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

resource whisperdeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAIResource && deployOpenAIWhisperModel) {
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

// Key Vault for storing the Functiona app's API key - the web app will get the function app's key from here when deployed
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      name: 'standard'
      family: 'A'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    publicNetworkAccess: storageServicesAndKVAllowPublicAccess ? 'Enabled' : 'Disabled'
    networkAcls: {
      defaultAction: 'Deny'
      // Only include VNET rules when using service endpoints
      virtualNetworkRules: backendServicesNetworkingType == 'ServiceEndpoint'
        ? [
            {
              id: '${vnet.id}/subnets/${webAppSubnetName}'
              ignoreMissingVnetServiceEndpoint: false
            }
          ]
        : []
      // Only include IP rules when using service endpoints and public access rules are enabled
      ipRules: (storageServicesAndKVAllowPublicAccess && !empty(storageServicesAndKVAllowedExternalIpsOrIpRanges))
        ? map(storageServicesAndKVAllowedExternalIpsOrIpRanges, ip => {
            value: ip
          })
        : []
    }
  }
}

resource privateKeyVaultDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: privateKeyVaultDnsZoneName
  location: 'global'
}

resource privateKeyVaultDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  parent: privateKeyVaultDnsZone
  name: '${privateKeyVaultDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

resource keyVaultPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (storageServicesAndKVNetworkingType == 'PrivateEndpoint') {
  name: '${keyVault.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${storageServicesAndKVSubnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: '${keyVault.name}-plsc'
        properties: {
          privateLinkServiceId: keyVault.id
          groupIds: ['vault']
        }
      }
    ]
  }

  resource keyVaultPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'keyVaultPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateKeyVaultDnsZone.id
          }
        }
      ]
    }
  }
}

resource keyVaultAccessPolicies 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  parent: keyVault
  name: 'add'
  properties: {
    accessPolicies: deployWebApp
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
  }
}

resource funcAppKvSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: funcAppKeyKvSecretName
  parent: keyVault
  properties: {
    value: listkeys('${functionApp.id}/host/default', '2022-03-01').functionKeys.default
  }
}

/// Function App
// Create a new Log Analytics workspace to back the Azure Application Insights instance
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
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

resource functionAppPlan 'Microsoft.Web/serverfarms@2024-04-01' = {
  name: functionAppPlanTokenName
  location: location
  properties: {
    reserved: true
  }
  sku: functionAppSkuProperties
  kind: 'linux'
}

// Populate environment variables for the function app for backend services,
// only including values for resources & endpoints that are deployed 
var optionalDeploymentFuncAppEnvVars = union(
  deployCosmosDB
    ? {
        CosmosDbConnectionSetting__accountEndpoint: cosmosDbAccount.properties.documentEndpoint
        COSMOSDB_DATABASE_NAME: cosmosDbDatabaseName
      }
    : {},
  deployContentUnderstandingMultiServicesResource
    ? {
        CONTENT_UNDERSTANDING_ENDPOINT: 'https://${contentUnderstandingTokenName}.services.ai.azure.com/'
      }
    : {},
  deployDocIntelResource
    ? {
        DOC_INTEL_ENDPOINT: docIntel.properties.endpoint
      }
    : {},
  deployLanguageResource
    ? {
        LANGUAGE_ENDPOINT: 'https://${languageTokenName}.cognitiveservices.azure.com/'
      }
    : {},
  deploySpeechResource
    ? {
        SPEECH_ENDPOINT: speech.properties.endpoint
      }
    : {},
  deployOpenAIResource
    ? {
        AOAI_ENDPOINT: azureopenai.properties.endpoint
      }
    : {},
  (deployOpenAIResource && deployOpenAILLMModel)
    ? {
        AOAI_LLM_DEPLOYMENT: openAILLMDeploymentName
      }
    : {},
  (deployOpenAIResource && deployOpenAIWhisperModel)
    ? {
        AOAI_WHISPER_DEPLOYMENT: openAIWhisperDeploymentName
      }
    : {}
)

// Create file share for function app
var functionContentShareName = 'function-content-share'
resource fileService 'Microsoft.Storage/storageAccounts/fileServices/shares@2022-05-01' = {
  name: '${storageAccountTokenName}/default/${functionContentShareName}'
  dependsOn: [
    storageAccount
  ]
}

resource functionApp 'Microsoft.Web/sites@2024-04-01' = {
  name: functionAppTokenName
  kind: 'functionapp,linux'
  location: location
  tags: union(tags, { 'azd-service-name': apiServiceName })
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publicNetworkAccess: functionAppAllowPublicAccess ? 'Enabled' : 'Disabled'
    httpsOnly: true
    serverFarmId: functionAppPlan.id
    clientAffinityEnabled: true
    siteConfig: {
      alwaysOn: functionAppUsePremiumSku ? true : false
      pythonVersion: '3.11'
      linuxFxVersion: 'python|3.11'
      cors: {
        allowedOrigins: [
          '*'
        ]
      }
      scmIpSecurityRestrictionsDefaultAction: 'Deny'
      scmIpSecurityRestrictionsUseMain: true // Use same IP restrictions for the SCM deployment site as the main site
      ipSecurityRestrictions: concat(
        // Allow access from the web app subnet when using service endpoints
        [
          {
            vnetSubnetResourceId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, webAppSubnetName)
            action: 'Allow'
            priority: 100
            name: 'Allow web app subnet'
          }
        ],
        // If: public network access is allowed and IPs are specified, Add IP rules and deny all other traffic
        functionAppAllowPublicAccess && !empty(functionAppAllowedExternalIpRanges)
          ? concat(
              map(range(0, length(functionAppAllowedExternalIpRanges)), i => {
                ipAddress: functionAppAllowedExternalIpRanges[i]
                action: 'Allow'
                priority: 200 + i
                name: 'External access ${i + 1}'
              }),
              [
                {
                  ipAddress: '0.0.0.0/0'
                  action: 'Deny'
                  priority: 2147483647
                  name: 'Deny all'
                }
              ]
            )
          : [],
        // OR: If public network access is allowed and no IPs are specified, allow all traffic
        functionAppAllowPublicAccess && empty(functionAppAllowedExternalIpRanges)
          ? [
              {
                ipAddress: '0.0.0.0/0'
                action: 'Allow'
                priority: 2147483647
                name: 'Allow all'
              }
            ]
          : []
      )
    }
  }
  resource functionSettings 'config@2024-04-01' = {
    name: 'appsettings'
    properties: union(functionAppConsumptionSettings, optionalDeploymentFuncAppEnvVars, {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      AzureWebJobsFeatureFlags: 'EnableWorkerIndexing'
      AzureWebJobsStorage__credential: 'managedIdentity'
      AzureWebJobsStorage__serviceUri: storageAccountBlobUri
      AzureWebJobsStorage__queueServiceUri: storageAccountQueueUri
      AzureWebJobsStorage__tableServiceUri: storageAccountTableUri
      WEBSITE_CONTENTSHARE: functionContentShareName
      WEBSITE_CONTENTAZUREFILECONNECTIONSTRING: storageAccountConnectionString
      WEBSITE_CONTENTOVERVNET: '1'
      WEBSITE_VNET_ROUTE_ALL: '1'
      APPLICATIONINSIGHTS_CONNECTION_STRING: appInsights.properties.ConnectionString
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
    })
  }
  dependsOn: [
    fileService
  ]
}

// Web & Function app shared private DNS zone
resource privateAppDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = if (functionAppNetworkingType == 'PrivateEndpoint' || webAppUsePrivateEndpoint) {
  name: privateAppDnsZoneName
  location: 'global'
}

resource privateAppDnsZoneLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = if (functionAppNetworkingType == 'PrivateEndpoint' || webAppUsePrivateEndpoint) {
  parent: privateAppDnsZone
  name: '${privateAppDnsZoneName}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// Function App Private Endpoint resources
resource privateEndpointFunctionApp 'Microsoft.Network/privateEndpoints@2024-05-01' = if (functionAppNetworkingType == 'PrivateEndpoint') {
  name: '${functionApp.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, functionAppPrivateEndpointSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyFunctionAppPrivateLinkConnection'
        properties: {
          privateLinkServiceId: functionApp.id
          groupIds: [
            'sites'
          ]
        }
      }
    ]
  }

  resource FunctionAppPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = {
    name: 'functionAppPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateAppDnsZone.id
          }
        }
      ]
    }
  }
}

// Create VNET integration
resource functionAppVirtualNetwork 'Microsoft.Web/sites/networkConfig@2024-04-01' = {
  parent: functionApp
  name: 'virtualNetwork'
  properties: {
    subnetResourceId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, functionAppSubnetName)
    swiftSupported: true
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

module functionAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = if (deployCosmosDB) {
  name: 'functionAppCosmosDbRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    principalId: functionApp.identity.principalId
    roleDefinitionId: cosmosDbDataContributorRoleDefinition.id
  }
}

// Demo app
resource webAppPlan 'Microsoft.Web/serverfarms@2024-04-01' = if (deployWebApp) {
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

// Populate environment variables for the demo app for backend services,
// only including values for resources & endpoints that are deployed 
var optionalDeploymentWebAppEnvVars = deployCosmosDB
  ? {
      COSMOSDB_ACCOUNT_ENDPOINT: cosmosDbAccount.properties.documentEndpoint
      COSMOSDB_DATABASE_NAME: cosmosDbDatabaseName
    }
  : {}

resource webApp 'Microsoft.Web/sites@2024-04-01' = if (deployWebApp) {
  name: webAppTokenName
  location: location
  tags: union(tags, { 'azd-service-name': webAppServiceName })
  kind: 'app,linux'
  properties: {
    publicNetworkAccess: webAppAllowPublicAccess ? 'Enabled' : 'Disabled'
    clientAffinityEnabled: true // If app plan capacity > 1, set this to True to ensure gradio works correctly.
    serverFarmId: webAppPlan.id
    httpsOnly: true
    siteConfig: {
      alwaysOn: true
      linuxFxVersion: 'python|3.11'
      ftpsState: 'Disabled'
      appCommandLine: 'python demo_app.py'
      minTlsVersion: '1.2'
      scmIpSecurityRestrictionsDefaultAction: 'Deny'
      scmIpSecurityRestrictionsUseMain: true // Use same IP restrictions for the SCM deployment site as the main site
      ipSecurityRestrictionsDefaultAction: 'Deny'
      ipSecurityRestrictions: concat(
        // Allow access from the web app subnet
        [
          {
            vnetSubnetResourceId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, webAppSubnetName)
            action: 'Allow'
            priority: 100
            name: 'Allow web app subnet'
          }
        ],
        // If: public network access is allowed and IPs are specified, Add IP rules and deny all other traffic
        webAppAllowPublicAccess && !empty(webAppAllowedExternalIpRanges)
          ? concat(
              map(range(0, length(webAppAllowedExternalIpRanges)), i => {
                ipAddress: webAppAllowedExternalIpRanges[i]
                action: 'Allow'
                priority: 200 + i
                name: 'External access ${i + 1}'
              }),
              [
                {
                  ipAddress: '0.0.0.0/0'
                  action: 'Deny'
                  priority: 2147483647
                  name: 'Deny all'
                }
              ]
            )
          : [],
        // OR: If public network access is allowed and no IPs are specified, allow all traffic
        webAppAllowPublicAccess && empty(webAppAllowedExternalIpRanges)
          ? [
              {
                ipAddress: '0.0.0.0/0'
                action: 'Allow'
                priority: 2147483647
                name: 'Allow all'
              }
            ]
          : []
      )
    }
  }
  identity: {
    type: 'SystemAssigned'
  }

  resource appSettings 'config@2024-04-01' = {
    name: 'appsettings'
    properties: union(optionalDeploymentWebAppEnvVars, {
      SCM_DO_BUILD_DURING_DEPLOYMENT: '1'
      FUNCTION_HOSTNAME: 'https://${functionApp.properties.defaultHostName}'
      FUNCTION_KEY: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=${funcAppKeyKvSecretName})'
      STORAGE_ACCOUNT_ENDPOINT: storageAccount.properties.primaryEndpoints.blob
      WEB_APP_USE_PASSWORD_AUTH: string(webAppUsePasswordAuth)
      WEB_APP_USERNAME: webAppUsername // Demo app username, no need for key vault storage
      WEB_APP_PASSWORD: webAppPassword // Demo app password, no need for key vault storage
      WEBSITE_VNET_ROUTE_ALL: '1' // Force all traffic through VNET
    })
  }
}

resource privateEndpointWebApp 'Microsoft.Network/privateEndpoints@2024-05-01' = if (webAppUsePrivateEndpoint) {
  name: '${webApp.name}-private-endpoint'
  location: location
  properties: {
    subnet: {
      id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, webAppPrivateEndpointSubnetName)
    }
    privateLinkServiceConnections: [
      {
        name: 'MyFunctionAppPrivateLinkConnection'
        properties: {
          privateLinkServiceId: webApp.id
          groupIds: [
            'sites'
          ]
        }
      }
    ]
  }

  resource WebAppPrivateDnsZoneGroup 'privateDnsZoneGroups@2024-05-01' = if (webAppUsePrivateEndpoint) {
    name: 'webAppPrivateDnsZoneGroup'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: 'config'
          properties: {
            privateDnsZoneId: privateAppDnsZone.id
          }
        }
      ]
    }
  }
}

// Create VNET integration
resource webAppVirtualNetwork 'Microsoft.Web/sites/networkConfig@2024-04-01' = {
  parent: webApp
  name: 'virtualNetwork'
  properties: {
    subnetResourceId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnet.name, webAppSubnetName)
    swiftSupported: true
  }
}

// Role assignments for the Web App's managed identity

// Add blob storage contributor role to web app (for uploading input data and triggering the function app)
resource webAppBlobContainerRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for containerName in blobContainerNames: {
    name: guid(storageAccount.id, containerName, 'WebAppBlobContainerRoleAssignment')
    scope: blobStorageContainer[indexOf(blobContainerNames, containerName)]
    properties: {
      roleDefinitionId: subscriptionResourceId(
        'Microsoft.Authorization/roleDefinitions',
        roleDefinitions.storageBlobDataContributor
      )
      principalId: webApp.identity.principalId
    }
  }
]

// Add CosmosDB data reader role to web app (for reading output data - e.g. processed documents)
module webAppCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = if (deployWebApp && deployCosmosDB) {
  name: 'webAppCosmosDbReaderRoleAssignment'
  scope: resourceGroup()
  params: {
    accountName: cosmosDbAccount.name
    principalId: webApp.identity.principalId
    roleDefinitionId: cosmosDbDataReaderRoleDefinition.id
  }
}

// Role assignments for additional identities (e.g. for local development)
module additionalIdentityStorageRoleAssignments 'storage-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: {
    name: guid(storageAccount.id, additionalRoleAssignmentIdentityId, 'StorageRoleAssignment')
    scope: resourceGroup()
    params: {
      storageAccountName: storageAccount.name
      principalId: additionalRoleAssignmentIdentityId
      roleDefintionIds: storageRoleDefinitionIds
    }
  }
]
module additionalIdentityCosmosDbRoleAssignment 'cosmosdb-account-role-assignment.bicep' = [
  for additionalRoleAssignmentIdentityId in additionalRoleAssignmentIdentityIds: if (deployCosmosDB) {
    name: guid(cosmosDbAccount.id, additionalRoleAssignmentIdentityId, 'CosmosDbRoleAssignment')
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
output storageAccountEndpolint string = storageAccount.properties.primaryEndpoints.blob
output storageAccountName string = storageAccount.name
output storageAccountBlobUri string = storageAccountBlobUri
output storageAccountQueueUri string = storageAccountQueueUri
output storageAccountTableUri string = storageAccountTableUri
output cosmosDbAccountEndpoint string = deployCosmosDB ? cosmosDbAccount.properties.documentEndpoint : ''
output cosmosDbAccountName string = deployCosmosDB ? cosmosDbAccount.name : ''
output cosmosDbDatabaseName string = deployCosmosDB ? cosmosDbDatabaseName : ''
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
