using Microsoft.AspNetCore.Antiforgery;

namespace Drumwise.API.Endpoints;

public static class SecurityEndpoints
{
    public static void MapSecurityEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var securityGroup = endpoints.MapGroup("/security");//.RequireAuthorization();
        
        securityGroup.MapGet("antiforgery/token", (IAntiforgery forgeryService, HttpContext context) =>
        {
            var tokens = forgeryService.GetAndStoreTokens(context);
            var xsrfToken = tokens.RequestToken!;
            return TypedResults.Content(xsrfToken, "text/plain");
        });
    }
}