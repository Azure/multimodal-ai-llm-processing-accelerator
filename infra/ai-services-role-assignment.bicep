metadata description = 'Assign RBAC resource-level roles to an Azure AI Service.'

@description('The name of the AI service resource to assign roles for.')
param aiServiceName string

@description('Id of the identity/principal to assign roles.')
param principalId string

@description('A list of role definition Ids to assign to the targeted principal.')
param roleDefinitionIds array

// Get a reference to the AI service
resource aiService 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: aiServiceName
}

resource aiServiceRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for roleDefinitionId in roleDefinitionIds: {
    name: guid(subscription().id, resourceGroup().id, aiService.id, roleDefinitionId, principalId)
    scope: aiService
    properties: {
      roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
      principalId: principalId
    }
  }
]
