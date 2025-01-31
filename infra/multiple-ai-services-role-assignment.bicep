metadata description = 'Assign RBAC resource-level roles to an Azure AI Service.'

@description('The name of the AI service resource to assign roles for.')
param aiServiceName string

@description('Id of the identity/principal to assign roles.')
param principalIds array

@description('A list of role definition Ids to assign to the targeted principal.')
param roleDefinitionIds array

module aiServicesRoleAssignments 'ai-services-role-assignment.bicep' = [
  for principalId in principalIds: {
    name: guid(subscription().id, resourceGroup().id, aiServiceName, principalId, 'MultiAiServicesRoleAssignment')
    params: {
      aiServiceName: aiServiceName
      principalId: principalId
      roleDefinitionIds: roleDefinitionIds
    }
  }
]
