using System.ComponentModel.Design.Serialization;
using System.Diagnostics;
using Drumwise.Application.Common.Interfaces;
using MediatR;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class PerformanceBahaviour<TRequest, TResponse>(
    Stopwatch timer,
    ILogger logger,
    IUser user,
    IIdentityService identityService)
    : IPipelineBehavior<TRequest, TResponse>
    where TRequest : notnull
{
    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        timer.Start();

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

            logger.Warn("Drumwise Long Running Request: {Name} ({ElapsedMilliseconds} milliseconds), {@UserId}, {@UserName}, {@Request}",
                requestName, elapsedMilliseconds, userId, userName, request);
        }

        return response;
    }
}