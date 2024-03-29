using System.ComponentModel.Design.Serialization;
using System.Diagnostics;
using Drumwise.Application.Common.Interfaces;
using MediatR;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class PerformanceBahaviour<TRequest, TResponse>(IUser user, IIdentityService identityService)
    : IPipelineBehavior<TRequest, TResponse> where TRequest : notnull
{
    private readonly ILogger _logger = LogManager.GetCurrentClassLogger();
    
    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        var timer  = Stopwatch.StartNew();

        var response = await next();

        timer.Stop();

        var elapsedMilliseconds = timer.ElapsedMilliseconds;

        if (elapsedMilliseconds > 500)
        {
            const string requestName = nameof(TRequest);
            var userId = user.Id ?? string.Empty;
            var userName = string.Empty;

            if (!string.IsNullOrEmpty(userId))
            {
                userName = await identityService.GetUserNameAsync(userId);
            }

            _logger.Warn("Drumwise Long Running Request: {Name} ({ElapsedMilliseconds} milliseconds), {@UserId}, {@UserName}, {@Request}",
                requestName, elapsedMilliseconds, userId, userName, request);
        }

        return response;
    }
}