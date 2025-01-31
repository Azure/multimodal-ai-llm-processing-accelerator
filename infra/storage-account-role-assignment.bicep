metadata description = 'Assign RBAC resource-level roles to an Azure Storage Account.'

@description('Name of the Storage account.')
param storageAccountName string

@description('Id of the identity/principal to assign roles to.')
param principalId string

@description('A list of role definition Ids to assign to the targeted principal in the context of the account.')
param roleDefintionIds array

// Get a reference to the storage account
resource storageAccount 'Microsoft.Storage/storageAccounts@2019-06-01' existing = {
  name: storageAccountName
}

resource storageAccountContributorRoles 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = [
  for roleDefinitionId in roleDefintionIds: {
    name: guid(storageAccount.id, roleDefinitionId, principalId)
    scope: storageAccount
    properties: {
      roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
      principalId: principalId
    }
  }
]
