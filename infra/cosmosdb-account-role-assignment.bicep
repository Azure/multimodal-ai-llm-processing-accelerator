metadata description = 'Assign RBAC resource-level role to Azure Cosmos DB for NoSQL.'

@description('Name of the Azure Cosmos DB for NoSQL account.')
param accountName string

@description('Id of the identity/principal to assign this role in the context of the account.')
param principalId string

@description('Id of the role definition to assign to the targeted principal in the context of the account.')
param roleDefinitionId string

// Get a reference to the CosmosDB account
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: accountName
}

resource assignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  parent: cosmosDbAccount
  name: guid(cosmosDbAccount.id, roleDefinitionId, principalId)
  properties: {
    roleDefinitionId: roleDefinitionId
    principalId: principalId
    scope: cosmosDbAccount.id
  }
}

output assignmentId string = assignment.id
