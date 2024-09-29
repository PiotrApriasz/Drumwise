using Drumwise.API.Endpoints;
using Drumwise.Infrastructure.Identity;
using SharpGrip.FluentValidation.AutoValidation.Endpoints.Extensions;

namespace Drumwise.API;

internal static class ApplicationEndpoints
{
    internal static void MapApplicationEndpoints(this WebApplication app)
    {
        var apiEndpoints = app.MapGroup(ApiPaths.MainPath).AddFluentValidationAutoValidation();
        
        apiEndpoints.MapIdentityApi<ApplicationUser>();
        apiEndpoints.MapAdditionalIdentityEndpoints();
        
        apiEndpoints.MapSecurityEndpoints();
        
        apiEndpoints.MapHomeworkEndpoints();
        
    }
}