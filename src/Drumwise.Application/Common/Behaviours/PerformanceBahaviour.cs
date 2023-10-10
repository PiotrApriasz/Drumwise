using System.ComponentModel.Design.Serialization;
using System.Diagnostics;
using Drumwise.Application.Common.Interfaces;
using MediatR;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class PerformanceBahaviour<TRequest, TResponse> : IPipelineBehavior<TRequest, TResponse> where TRequest : notnull
{
    private readonly Stopwatch _timer;
    private readonly ILogger _logger;
    private readonly IUser _user;
    private readonly IIdentityService _identityService;

    public PerformanceBahaviour(Stopwatch timer, ILogger logger, IUser user, IIdentityService identityService)
    {
        _timer = timer;
        _logger = logger;
        _user = user;
        _identityService = identityService;
    }

    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        _timer.Start();

        var response = await next();

        _timer.Stop();

        var elapsedMilliseconds = _timer.ElapsedMilliseconds;

        if (elapsedMilliseconds > 500)
        {
            const string requestName = nameof(TRequest);
            var userId = _user.Id ?? string.Empty;
            var userName = string.Empty;

            if (!string.IsNullOrEmpty(userId))
            {
                userName = await _identityService.GetUserNameAsync(userId);
            }

            _logger.Warn("Drumwise Long Running Request: {Name} ({ElapsedMilliseconds} milliseconds), {@UserId}, {@UserName}, {@Request}",
                requestName, elapsedMilliseconds, userId, userName, request);
        }

        return response;
    }
}