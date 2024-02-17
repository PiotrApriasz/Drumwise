using Drumwise.Application.Common.Models.Identity;

namespace Drumwise.API.Middleware;

public class ClientApiKeyAuthenticationMiddleware : IMiddleware
{
    private const string ApiKeyHeaderName = "X-ClientAPI-Key";
    
    public async Task InvokeAsync(HttpContext context, RequestDelegate next)
    {
        if (!context.Request.Headers.TryGetValue(ApiKeyHeaderName, out var extractedApiKey))
        {
            context.Response.StatusCode = 401;
            await context.Response.WriteAsync("Api Key was not provided.");
            return;
        }

        var appSettings = context.RequestServices.GetRequiredService<IConfiguration>();
        var apiKeys = appSettings.GetSection("ClientApiKeys").Get<List<ClientApiKey>>();

        var apiKey = apiKeys!.FirstOrDefault(x => x.Key == extractedApiKey);

        if (apiKey == null)
        {
            context.Response.StatusCode = 401;
            await context.Response.WriteAsync("Unauthorized client.");
            return;
        }
        
        context.Items["ClientName"] = apiKey.Name;
        context.Items["ClientAddress"] = apiKey.Address;

        await next(context);
    }
}